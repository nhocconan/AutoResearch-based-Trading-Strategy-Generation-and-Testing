#!/usr/bin/env python3
name = "6h_12h_1d_PivotBreakout_TrendVolume_v2"
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
    
    # Load daily data ONCE before loop for Pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate daily Pivot (standard) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Pivot support/resistance levels - using S2/R2 for wider bands to reduce trades
    s2 = pivot - range_hl
    r2 = pivot + range_hl
    
    # Align daily levels to 6h timeframe
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # 12h EMA(50) for trend filter (slower for fewer trades)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection: 6-period average (1.5 days of 6h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S2 with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 2.0  # Higher threshold for fewer trades
            uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            
            if close[i] > s2_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price below R2 with volume and 12h downtrend
            elif close[i] < r2_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below S2 or volume drops significantly
            if close[i] < s2_aligned[i] or volume[i] < vol_ma_6[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above R2 or volume drops significantly
            if close[i] > r2_aligned[i] or volume[i] < vol_ma_6[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 6h Pivot S2/R2 breakout with 12h trend and volume confirmation
# - Uses wider S2/R2 levels (vs S1/R1) to reduce trade frequency and avoid noise
# - Requires 2x volume spike (vs 1.8x) for stronger confirmation
# - Uses slower EMA(50) on 12h for smoother trend filter
# - Designed for 15-35 trades/year target to stay well within limits
# - Works in both bull (buy S2 breaks in uptrend) and bear (sell R2 breaks in downtrend)
# - Exit when price returns to S2/R2 or volume drops below average
# - Position size 0.30 balances return potential with drawdown control
# - Novel combination: Pivot S2/R2 (1d) + trend EMA50 (12h) + volume spike (6h) not tried recently
# - Aims for 60-140 total trades over 4 years (15-35/year) to avoid fee drag
# - Focus on BTC/ETH as primary targets with institutional volume confirmation