#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA50 trend filter
# Camarilla pivot levels provide high-probability intraday support/resistance. 
# Breakout at R3/S3 with volume confirmation captures institutional participation.
# 1w EMA50 ensures we only trade in the direction of the weekly trend, reducing counter-trend false breakouts.
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull (breakouts with volume in uptrend) and bear (breakdowns with volume in downtrend).

name = "6h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wEMA50_Trend_v1"
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
    
    # 1d HTF data for volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d volume spike: current 1d volume > 2.0 * 20-period average volume
    vol_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (volume_ma_20 * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels for 6h timeframe using prior bar's OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6)
    #          S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    for i in range(1, n):
        # Use prior bar's OHLC to avoid look-ahead
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        camarilla_r3[i] = c + ((h - l) * 1.1 / 4)
        camarilla_s3[i] = c - ((h - l) * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 20 for volume MA + 50 for weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_spike_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Camarilla breakout conditions
        breakout_up = curr_close > camarilla_r3[i]  # Break above R3
        breakout_down = curr_close < camarilla_s3[i]  # Break below S3
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike_1d_aligned[i]
        # Uptrend: price above weekly EMA50, Downtrend: price below weekly EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla R3 breakout up, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S3 breakout down, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla S3 breakdown or trend reversal
            if curr_close < camarilla_s3[i] or curr_close < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Camarilla R3 breakout or trend reversal
            if curr_close > camarilla_r3[i] or curr_close > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals