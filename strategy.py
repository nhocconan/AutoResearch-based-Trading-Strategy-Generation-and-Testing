#!/usr/bin/env python3

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous day
    # We'll use daily high/low/close to compute today's Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R3, S3 levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 = close + (high - low) * 1.1 / 4
    # Camarilla S3 = close - (high - low) * 1.1 / 4
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 1.5x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~6 hours for 4h to reduce trades
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Align 12h EMA50 to 4h timeframe
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
        
        # Determine 12h trend direction
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_12h_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above Camarilla R3 in 12h uptrend with volume
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_12h_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below Camarilla S3 in 12h downtrend with volume
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_12h_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Break below Camarilla S3 or trend change
            if (close[i] < camarilla_s3_aligned[i]) or not trend_12h_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Break above Camarilla R3 or trend change
            if (close[i] > camarilla_r3_aligned[i]) or not trend_12h_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout on 4h with 12h EMA50 trend filter and volume confirmation captures institutional breakouts in both bull and bear markets. The 4h timeframe balances trade frequency and responsiveness, while Camarilla levels derived from daily price action provide key support/resistance levels watched by institutions. Volume confirmation ensures genuine breakouts, and the 12h trend filter avoids counter-trend trades. Target: 25-35 trades/year. Works in bull markets by buying R3 breakouts in uptrends and in bear markets by selling S3 breakdowns in downtrends. This avoids overtrading by requiring multiple confirmations and cooldown periods.