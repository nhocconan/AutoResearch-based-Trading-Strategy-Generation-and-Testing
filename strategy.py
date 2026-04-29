#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Camarilla Pivot Breakout with 1d EMA50 trend filter and volume confirmation
# Uses weekly Camarilla pivot levels (R3/S3, R4/S4) from prior week for structure.
# Long when price breaks above weekly R4 with volume > 2x average and close > 1d EMA50.
# Short when price breaks below weekly S4 with volume > 2x average and close < 1d EMA50.
# Exit when price returns to weekly R3/S3 (mean reversion) or opposite Camarilla level breached.
# Weekly pivot provides structural filters reducing whipsaw in ranging markets.
# Volume confirmation ensures institutional participation.
# 1d EMA50 aligns with daily trend to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_WeeklyCamarilla_R4S4_Breakout_1dEMA50_Volume_v1"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly Camarilla pivots from prior week's OHLC
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Camarilla formulas (based on prior week's range)
    # H = high, L = low, C = close of prior weekly bar
    H = df_1w['high'].values
    L = df_1w['low'].values
    C = df_1w['close'].values
    
    # Camarilla levels
    R4 = C + ((H - L) * 1.1 / 2)
    R3 = C + ((H - L) * 1.1 / 4)
    S3 = C - ((H - L) * 1.1 / 4)
    S4 = C - ((H - L) * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe (use prior week's levels)
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 1d EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        curr_R4 = R4_aligned[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        curr_S4 = S4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above weekly R4 with close > 1d EMA50
                if curr_close > curr_R4 and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below weekly S4 with close < 1d EMA50
                elif curr_close < curr_S4 and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price returns to weekly R3 (mean reversion) or breaks below S3
            if curr_close <= curr_R3 or curr_close < curr_S3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to weekly S3 (mean reversion) or breaks above R3
            if curr_close >= curr_S3 or curr_close > curr_R3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals