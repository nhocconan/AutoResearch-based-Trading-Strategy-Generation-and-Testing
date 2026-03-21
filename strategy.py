#!/usr/bin/env python3
"""
EXPERIMENT #024 - Daily Supertrend + Weekly HMA Filter + RSI Pullback (1d primary, 1w HTF)
==========================================================================================
Hypothesis: Daily charts have significantly less noise than intraday timeframes. By using
1d as primary with 1w HTF trend filter, we can capture major crypto trends while avoiding
whipsaws. Supertrend provides clear trend direction, RSI pullback entries improve risk/reward,
and weekly HMA ensures we only trade with the major trend.

Key features:
- Primary TF: 1d (required for this experiment)
- HTF filter: 1w HMA(21) - price must be above for longs, below for shorts
- Trend indicator: Supertrend(10, 3) on daily
- Entry trigger: RSI(14) pullback to 40-55 zone in uptrend (55-70 for shorts)
- Volume confirmation: volume > 20-day SMA
- Stoploss: 3.0*ATR(14) trailing (wider for daily timeframe)
- Position sizing: 0.30 discrete (30% of capital)
- Take profit: Reduce to half at 2R profit

Why this differs from failures:
- Most failed strategies used intraday TFs (15m-4h) with too much noise
- Daily naturally filters out false signals - fewer but higher quality trades
- Weekly HTF is stronger filter than 1d/4h used in failed strategies
- Supertrend + RSI pullback has better win rate than breakout strategies (#016, #017, #020)
- Conservative position sizing (0.30 max) protects against 2022-style crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_weekly_1d_1w_v1"
timeframe = "1d"
leverage = 1.0


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate HL2 (median price)
    hl2 = (high + low) / 2.0
    
    # Calculate upper and lower bands
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Calculate Supertrend values
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = uptrend, -1 = downtrend
    
    supertrend[0] = upper_band[0]
    
    for i in range(1, n):
        if close[i - 1] > supertrend[i - 1]:
            # Was in uptrend
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
        else:
            # Was in downtrend
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
    
    return supertrend, trend


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


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for confirmation"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    supertrend, supertrend_trend = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.30  # Base position size (30% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
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
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # ===== WEEKLY TREND FILTER (MAJOR TREND DIRECTION) =====
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # ===== DAILY SUPERTREND TREND =====
        daily_trend = supertrend_trend[i]  # 1 = uptrend, -1 = downtrend
        
        # ===== VOLUME CONFIRMATION =====
        volume_ok = volume[i] > vol_sma[i]
        
        # ===== RSI PULLBACK ENTRY ZONE =====
        # Long: RSI pulled back to 40-55 in uptrend
        # Short: RSI rallied to 55-70 in downtrend
        rsi_long_pullback = 40 <= rsi[i] <= 55
        rsi_short_pullback = 55 <= rsi[i] <= 70
        
        # ===== TREND ALIGNMENT =====
        # Only trade when weekly and daily trends agree
        trends_aligned = (weekly_trend == daily_trend)
        
        # Determine target signal
        target_signal = 0.0
        
        if trends_aligned and weekly_trend == 1 and daily_trend == 1 and rsi_long_pullback and volume_ok:
            target_signal = SIZE
        elif trends_aligned and weekly_trend == -1 and daily_trend == -1 and rsi_short_pullback and volume_ok:
            target_signal = -SIZE
        
        # ===== STOPLOSS AND TAKE PROFIT LOGIC =====
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 3.0 * entry_atr
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 3*ATR)
                if not profit_target_hit and entry_atr > 0:
                    if close[i] >= entry_price + 6.0 * entry_atr:
                        take_profit_triggered = True
                
                # Check trend reversal (Supertrend flipped or weekly turned bearish)
                if daily_trend == -1 or weekly_trend == -1:
                    trend_reversal = True
                    
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 3.0 * entry_atr
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit and entry_atr > 0:
                    if close[i] <= entry_price - 6.0 * entry_atr:
                        take_profit_triggered = True
                
                # Check trend reversal
                if daily_trend == 1 or weekly_trend == 1:
                    trend_reversal = True
        
        # Apply signals based on conditions
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
            
        elif trend_reversal:
            # Exit on major trend reversal
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            
        else:
            # Check for new entry or maintain position
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
                # Maintain existing position
                if profit_target_hit:
                    signals[i] = HALF_SIZE * position_side
                else:
                    signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals