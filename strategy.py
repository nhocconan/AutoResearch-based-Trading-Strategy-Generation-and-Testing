#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Breakout_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly Donchian Channel (20-week period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling high/low for 20 periods
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (only use completed weekly bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === Weekly EMA20 for trend filter ===
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Daily volume filter: current volume > 20-day average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR for volatility filter ===
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(atr10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions: Donchian breakout with trend and volume confirmation
            long_cond = (close[i] > donchian_high_aligned[i] and 
                        close[i] > ema20_1w_aligned[i] and
                        volume[i] > vol_ma20[i])
            
            short_cond = (close[i] < donchian_low_aligned[i] and 
                         close[i] < ema20_1w_aligned[i] and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below weekly EMA20 or ATR-based stop
            if close[i] < ema20_1w_aligned[i] or close[i] < (donchian_high_aligned[i] - 2.0 * atr10[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above weekly EMA20 or ATR-based stop
            if close[i] > ema20_1w_aligned[i] or close[i] > (donchian_low_aligned[i] + 2.0 * atr10[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout strategy with trend filter (weekly EMA20) and volume confirmation.
# Works in both bull and bear markets by capturing breakouts in the direction of the weekly trend.
# Uses daily timeframe for execution but relies on weekly structure for signal generation.
# Targets 30-100 trades over 4 years (7-25/year) to minimize fee drag. Uses discrete sizing (0.25).
# Includes ATR-based stop loss to manage risk. Designed for BTC/ETH where weekly structure
# remains relevant even during ranging periods.