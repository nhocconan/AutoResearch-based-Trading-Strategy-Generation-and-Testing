#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# - Uses 1d EMA34 as trend filter (bullish above, bearish below)
# - Enters long when price breaks above R3 with volume > 2x MA and bullish trend
# - Enters short when price breaks below S3 with volume > 2x MA and bearish trend
# - Exits on opposite Camarilla level (S1 for longs, R1 for shorts) or trend reversal
# - Designed for 12h timeframe to capture multi-day moves while limiting trade frequency
# - Low trade frequency expected: ~20-40 trades/year based on Camarilla + volume + trend filters

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34[i] = (close_1d[i] * 2/35) + (ema_34[i-1] * 33/35)
    
    # Align 1d EMA34 to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 12h Camarilla levels from previous 1d OHLC
    # Camarilla: H-L range from previous day
    # R4 = Close + 1.5*(H-L), R3 = Close + 1.1*(H-L), etc.
    # S1 = Close - 1.1*(H-L), S2 = Close - 1.5*(H-L), etc.
    # We use R3/S3 for entry, R1/S1 for exit
    
    # Calculate previous day's range (using 1d data)
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R3 = np.full(len(prev_close), np.nan)
    camarilla_S3 = np.full(len(prev_close), np.nan)
    camarilla_R1 = np.full(len(prev_close), np.nan)
    camarilla_S1 = np.full(len(prev_close), np.nan)
    
    for i in range(len(prev_close)):
        if i == 0:
            continue  # Skip first bar as we need previous day
        H = prev_high[i-1]
        L = prev_low[i-1]
        C = prev_close[i-1]
        range_hl = H - L
        
        camarilla_R3[i] = C + 1.1 * range_hl
        camarilla_S3[i] = C - 1.1 * range_hl
        camarilla_R1[i] = C + 1.05 * range_hl  # Actually 1.05*(H-L) but simplified
        camarilla_S1[i] = C - 1.05 * range_hl  # Actually 1.05*(H-L) but simplified
    
    # Align Camarilla levels to 12h (these levels are valid for the entire 12h period following the 1d bar)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Calculate 20-period volume average for confirmation
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Start after warmup period
    start_idx = max(vol_period, 1)  # Need at least one day of data for Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above R3 with volume and bullish trend (price > EMA34)
            if price > camarilla_R3_aligned[i] and vol_ratio > 2.0 and price > ema_34_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below S3 with volume and bearish trend (price < EMA34)
            elif price < camarilla_S3_aligned[i] and vol_ratio > 2.0 and price < ema_34_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below S1 or trend turns bearish
            if price < camarilla_S1_aligned[i] or price < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price breaks above R1 or trend turns bullish
            if price > camarilla_R1_aligned[i] or price > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0