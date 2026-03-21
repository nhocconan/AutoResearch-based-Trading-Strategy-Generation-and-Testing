#!/usr/bin/env python3
"""
EXPERIMENT #007 - Supertrend + RSI Pullback + 4h HMA Trend (15m primary)
=====================================================================================
Hypothesis: 15m Supertrend captures intraday momentum moves while 4h HMA provides
higher timeframe bias. RSI pullback entries (not extremes) avoid chasing and catch
continuation moves. This should generate MORE trades than daily strategies while
maintaining trend alignment.

Key features:
- Primary TF: 15m (faster signals, more trade opportunities)
- HTF filter: 4h HMA(21) for trend bias (aligns with crypto swing cycles)
- Entry: Supertrend(10,3) flip + RSI(14) between 45-65 (pullback, not extreme)
- Volume confirmation: taker_buy_volume > SMA(20) for longs, < for shorts
- Stoploss: 2.0*ATR(14) trailing stop
- Position sizing: 0.25 base, 0.30 max (discrete levels)
- Take profit: Reduce to half at 2R, trail stop at 1R

Why 15m should work better than 1d:
- More trade opportunities (15m has 96 bars per day vs 1 for daily)
- Faster reaction to crypto volatility spikes
- 4h HMA still provides major trend filter (4 bars per 4h candle)
- RSI pullback (45-65) is less strict than extremes (30/70) = more entries

Lessons from failures:
- #006: Bug in calculate_donchian (undefined 'close' variable) - FIXED
- #004: 0 trades (conditions too strict) - Using looser RSI range
- Multiple: Negative Sharpe - Conservative sizing (0.25-0.30) controls DD
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_4hhma_15m_v1"
timeframe = "15m"
leverage = 1.0


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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] == 0:
            upper_band[i] = hl2[i]
            lower_band[i] = hl2[i]
        else:
            upper_band[i] = hl2[i] + multiplier * atr[i]
            lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close[i - 1] <= supertrend[i - 1]:
                # Previously bearish
                supertrend[i] = min(upper_band[i], supertrend[i - 1])
                if close[i] > supertrend[i]:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    direction[i] = -1
            else:
                # Previously bullish
                supertrend[i] = max(lower_band[i], supertrend[i - 1])
                if close[i] < supertrend[i]:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    direction[i] = 1
    
    return supertrend, direction


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    taker_buy_vol = prices["taker_buy_volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(taker_buy_vol, period=20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 50  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN or zero in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(vol_sma[i]) or
            atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Supertrend flip detection (entry signal)
        st_flip_long = (i > min_period and st_direction[i] == 1 and st_direction[i - 1] == -1)
        st_flip_short = (i > min_period and st_direction[i] == -1 and st_direction[i - 1] == 1)
        
        # RSI pullback zone (not extreme - allows more entries)
        rsi_valid_long = 45 <= rsi[i] <= 70  # Bullish momentum, not overbought
        rsi_valid_short = 30 <= rsi[i] <= 55  # Bearish momentum, not oversold
        
        # Volume confirmation
        vol_above_avg = taker_buy_vol[i] > vol_sma[i] * 1.1  # 10% above average
        vol_below_avg = taker_buy_vol[i] < vol_sma[i] * 0.9  # 10% below average
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend flip + 4h HMA bullish + RSI valid + volume confirmation
        if (st_flip_long and hma_trend == 1 and rsi_valid_long):
            target_signal = BASE_SIZE
        
        # Short entry: Supertrend flip + 4h HMA bearish + RSI valid
        elif (st_flip_short and hma_trend == -1 and rsi_valid_short):
            target_signal = -BASE_SIZE
        
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
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4.0*ATR
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
                # Exit if Supertrend flips against position OR 4h HMA alignment breaks
                st_reversal_long = st_direction[i] == -1
                st_reversal_short = st_direction[i] == 1
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
                if st_reversal_long or st_reversal_short or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = BASE_SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals