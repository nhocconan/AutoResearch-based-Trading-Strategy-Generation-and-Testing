#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Power_Zone_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Elder Ray (Bull/Bear Power) from 1d ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 of close for Elder Ray calculation
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 1d EMA20 for trend filter ===
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # === 6h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d ATR for volatility filter ===
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for EMA20 and ATR10
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ma20[i]) or 
            np.isnan(atr10_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions: Elder Ray alignment + trend filter + volume
            # Long: Bull Power > 0 AND price above EMA20 (uptrend) AND volume confirmation
            long_cond = (bull_power_6h[i] > 0 and 
                        close[i] > ema20_1d_aligned[i] and
                        volume[i] > vol_ma20[i])
            
            # Short: Bear Power < 0 AND price below EMA20 (downtrend) AND volume confirmation
            short_cond = (bear_power_6h[i] < 0 and 
                         close[i] < ema20_1d_aligned[i] and
                         volume[i] > vol_ma20[i])
            
            # Additional volatility filter: avoid low volatility environments
            vol_filter = atr10_1d_aligned[i] > np.nanmedian(atr10_1d_aligned[max(0, i-50):i+1])
            
            if long_cond and vol_filter:
                signals[i] = 0.25
                position = 1
            elif short_cond and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power turns negative OR price breaks below EMA20
            if bear_power_6h[i] < 0 or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power turns positive OR price breaks above EMA20
            if bull_power_6h[i] > 0 or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) from daily timeframe captures institutional
# buying/selling pressure. Combined with EMA20 trend filter and volume confirmation,
# this identifies high-probability trend continuation entries. Works in bull markets
# via Bull Power > 0 + uptrend, and in bear markets via Bear Power < 0 + downtrend.
# Volume filter ensures participation, ATR filter avoids low-volatility whipsaws.
# Target: 50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to
# minimize fee churn. Uses daily Elder Ray as the primary signal generator.