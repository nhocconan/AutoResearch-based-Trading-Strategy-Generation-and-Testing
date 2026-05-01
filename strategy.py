#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide precise intraday support/resistance. R3/S3 are strong breakout levels.
# 1d EMA34 ensures alignment with the daily trend. Volume spike confirms institutional participation.
# Works in bull (breakouts with volume) and bear (volatility expansion after consolidation).
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) calculation
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 1d bar (using 4h data to approximate)
    # We need prior day's high, low, close for Camarilla calculation
    # Resample to daily using actual Binance daily data from get_htf_data
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # But we need to use the prior completed day's values
    # Since we have df_1d from get_htf_data, we can calculate levels there
    
    # Calculate Camarilla levels on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, S3 (using prior day's values to avoid look-ahead)
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (with 1-bar delay for completed day)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 34 for EMA + 20 for volume MA + buffer
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price above/below 1d EMA34
        trend_up = curr_close > ema_34_1d_aligned[i]
        trend_down = curr_close < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > camarilla_r3_1d_aligned[i-1]  # Break above R3
        breakout_down = curr_close < camarilla_s3_1d_aligned[i-1]  # Break below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla R3 breakout up, volume spike, uptrend
            if breakout_up and vol_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S3 breakout down, volume spike, downtrend
            elif breakout_down and vol_spike and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla S3 breakdown or trend reversal
            if curr_close < camarilla_s3_1d_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Camarilla R3 breakout or trend reversal
            if curr_close > camarilla_r3_1d_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals