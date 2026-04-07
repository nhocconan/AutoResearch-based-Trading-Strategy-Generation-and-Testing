#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + Weekly Trend + Volume Confirmation
# Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
# combined with weekly trend filter and volume confirmation provides high-probability 
# entries. Works in both bull and bear markets by trading with weekly trend direction 
# and using pivot levels as dynamic support/resistance. Targets 15-25 trades/year 
# with strict entry criteria to avoid overtrading.

name = "6h_camarilla_pivot_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA50 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily data for Camarilla pivots (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1 / 2)
    # R3 = C + ((H - L) * 1.1 / 4)
    # S3 = C - ((H - L) * 1.1 / 4)
    # S4 = C - ((H - L) * 1.1 / 2)
    
    # Get previous day's OHLC values
    prev_day_high = df_1d['high'].values
    prev_day_low = df_1d['low'].values
    prev_day_close = df_1d['close'].values
    
    # Calculate pivot levels
    camarilla_pp = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    camarilla_range = prev_day_high - prev_day_low
    camarilla_r4 = prev_day_close + (camarilla_range * 1.1 / 2.0)
    camarilla_r3 = prev_day_close + (camarilla_range * 1.1 / 4.0)
    camarilla_s3 = prev_day_close - (camarilla_range * 1.1 / 4.0)
    camarilla_s4 = prev_day_close - (camarilla_range * 1.1 / 2.0)
    
    # Align to 6h timeframe (these levels are valid for the entire day)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 20-period SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for volume SMA
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(camarilla_r4_6h[i]) or 
            np.isnan(camarilla_r3_6h[i]) or 
            np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(camarilla_s4_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA50 OR reaches S3 (take profit)
            if close[i] < ema50_6h[i] or close[i] <= camarilla_s3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA50 OR reaches R3 (take profit)
            if close[i] > ema50_6h[i] or close[i] >= camarilla_r3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above R4 with volume confirmation AND uptrend
            if (close[i] > camarilla_r4_6h[i] and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume confirmation AND downtrend
            elif (close[i] < camarilla_s4_6h[i] and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals