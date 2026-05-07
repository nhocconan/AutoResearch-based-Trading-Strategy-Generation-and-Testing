#!/usr/bin/env python3
name = "6h_VolumeSpike_CamarillaR3S3_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # Camarilla: H' = (H-L)*1.1/12 + C, L' = C - (H-L)*1.1/12
    # R3 = C + (H-L)*1.1/6, S3 = C - (H-L)*1.1/6
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:  # Need at least 2 days to calculate
            continue
        # Previous day's OHLC
        ph = df_1d['high'].values[i-1]
        pl = df_1d['low'].values[i-1]
        pc = df_1d['close'].values[i-1]
        rng = ph - pl
        if rng <= 0:
            continue
        camarilla_r3[i] = pc + (rng * 1.1 / 6)
        camarilla_s3[i] = pc - (rng * 1.1 / 6)
        camarilla_r4[i] = pc + (rng * 1.1 / 2)
        camarilla_s4[i] = pc - (rng * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (using previous day's values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume spike filter: current volume > 2.0x 20-period average (for 6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours for 6h to reduce trades
    
    start_idx = max(100, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla R3 with volume spike in uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_up[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume spike in downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_down[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below Camarilla S3 or trend changes
            if close[i] < camarilla_s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above Camarilla R3 or trend changes
            if close[i] > camarilla_r3_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with volume spike and 1d EMA34 trend filter on 6h timeframe.
# Long when price breaks above Camarilla R3 with volume spike in uptrend.
# Short when price breaks below Camarilla S3 with volume spike in downtrend.
# Uses 6h timeframe to balance trade frequency and capture meaningful trends.
# Volume spike filter (2x 20-period average) ensures momentum confirmation.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Based on top-performing pattern from DB: Camarilla breakout + volume + trend filter.