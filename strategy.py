# 1
#!/usr/bin/env python3
name = "6h_LiquidityPool_Sweep_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE for liquidity pool levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly liquidity pool: previous week's high and low (swing extremes)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Align weekly liquidity levels to 6h
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_week_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_week_low)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: sweep below weekly low with rejection + volume + uptrend
            sweep_low = low[i] <= weekly_low_aligned[i]
            close_above_low = close[i] > weekly_low_aligned[i]  # rejection
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if sweep_low and close_above_low and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: sweep above weekly high with rejection + volume + downtrend
            elif high[i] >= weekly_high_aligned[i] and close[i] < weekly_high_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks above weekly high or volume drops
            if close[i] > weekly_high_aligned[i] or volume[i] < vol_ma_4[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks below weekly low or volume drops
            if close[i] < weekly_low_aligned[i] or volume[i] < vol_ma_4[i] * 1.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Liquidity Pool Sweep with 1d trend and volume confirmation
# - Price often sweeps weekly highs/lows to trigger stops before reversing
# - Long when price sweeps below weekly low but closes back above it (bullish rejection)
# - Short when price sweeps above weekly high but closes back below it (bearish rejection)
# - Requires volume spike (2.0x average) to confirm institutional activity
# - Uses daily EMA(34) trend filter to align with higher timeframe momentum
# - Works in bull markets (buy sweeps of weekly lows in uptrend) and bear markets (sell sweeps of weekly highs in downtrend)
# - Exit when price breaks the weekly extreme in the direction of the trend or volume weakens
# - Novel combination: Weekly liquidity pools (1w) + trend (1d) + volume spike (6h)
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Position size 0.25 manages risk while allowing meaningful returns