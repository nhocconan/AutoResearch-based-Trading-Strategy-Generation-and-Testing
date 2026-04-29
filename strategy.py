#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivots from daily timeframe provide statistically significant support/resistance levels
# Breakouts at R3/S3 with continuation through R4/S4 capture strong momentum moves
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaws
# Volume > 2.0x 20-period average confirms institutional participation
# Discrete sizing (0.25) minimizes fee churn; target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear via trend filter and volume confirmation

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_hl * 1.1 / 4.0)  # ~1.1/4 = 0.275
    R4 = pivot + (range_hl * 1.1 / 2.0)  # ~1.1/2 = 0.55
    S3 = pivot - (range_hl * 1.1 / 4.0)
    S4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # warmup for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 with continuation through R4
                # and price above daily EMA34 (uptrend)
                if (curr_high > R3_aligned[i] and curr_close > R4_aligned[i] and 
                    curr_close > curr_ema_34_1d):
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 with continuation through S4
                # and price below daily EMA34 (downtrend)
                elif (curr_low < S3_aligned[i] and curr_close < S4_aligned[i] and 
                      curr_close < curr_ema_34_1d):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below R3 (failed breakout) or reverses below S3
            if curr_close < R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above S3 (failed breakdown) or reverses above R3
            if curr_close > S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals