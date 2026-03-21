#!/usr/bin/env python3
"""
EXPERIMENT #025 - 15m Supertrend + Volume Confirmation + HTF Trend Filter
================================================================================
Hypothesis: 15m timeframe captures intraday momentum while 1h HMA filters major 
trend direction. Adding volume confirmation on Supertrend flips reduces false 
breakouts. ADX regime filter ensures we only trade in trending conditions. 
Dynamic position sizing based on ATR volatility controls drawdown.

Key features:
- Primary TF: 15m (faster entries than 4h strategies)
- HTF filter: 1h HMA(21) for trend direction (align_htf_to_ltf for proper shift)
- Trend: Supertrend(10, 3) on 15m for entry timing
- Entry confirmation: Volume > 1.5x 20-bar MA + ADX(14) > 25
- Regime: BB Width expanding OR ADX > 25 (trending market)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25 base, reduced to 0.20 when ATR volatility high
- Take profit: Reduce to half at 2R profit, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_volume_adx_15m_1h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper_band[0]
    
    for i in range(1, n):
        if close[i - 1] <= supertrend[i - 1]:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
    
    return supertrend, direction


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di


def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1h HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Calculate BB Width rolling mean for expansion detection
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    REDUCED_SIZE = 0.20  # Reduced size when volatility high
    HALF_SIZE = BASE_SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1h_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or 
            np.isnan(bb_width[i]) or np.isnan(vol_ma[i]) or 
            atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1h trend filter (HTF) - only trade in direction of HTF trend
        htf_trend = 1 if close[i] > hma_1h_aligned[i] else -1
        
        # 15m Supertrend direction
        st_trend = int(st_direction[i])
        
        # Volume confirmation: current volume > 1.5x 20-bar MA
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # ADX regime filter: ADX > 25 indicates trending market
        adx_trending = adx[i] > 25
        
        # BB Width expansion: current width > 20-bar MA of width
        bb_expanding = bb_width[i] > bb_width_ma[i]
        
        # Regime valid: either ADX trending OR BB expanding
        regime_valid = adx_trending or bb_expanding
        
        # RSI filter: avoid extreme overbought/oversold for entries
        rsi_valid_long = rsi[i] < 70  # Not overbought for long entry
        rsi_valid_short = rsi[i] > 30  # Not oversold for short entry
        
        # Dynamic position sizing based on ATR volatility
        # If ATR% > 3%, reduce position size
        atr_pct = atr[i] / close[i] * 100
        current_size = REDUCED_SIZE if atr_pct > 3.0 else BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend bullish + HTF trend bullish + Volume confirmed + Regime valid + RSI valid
        if st_trend == 1 and htf_trend == 1 and volume_confirmed and regime_valid and rsi_valid_long:
            target_signal = current_size
        
        # Short entry: Supertrend bearish + HTF trend bearish + Volume confirmed + Regime valid + RSI valid
        elif st_trend == -1 and htf_trend == -1 and volume_confirmed and regime_valid and rsi_valid_short:
            target_signal = -current_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.0*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and st_trend == -1:
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                elif position_side == -1 and st_trend == 1:
                    # Trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = current_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals