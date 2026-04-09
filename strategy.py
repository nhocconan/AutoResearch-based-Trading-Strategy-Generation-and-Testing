#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter
# Uses daily Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# Only takes trades aligned with 1week EMA200 trend (bullish above, bearish below)
# Volume confirmation (>1.5x 20-period average) to avoid false breakouts
# Position size 0.25 to manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag
# Works in both bull/bear: 1w EMA200 trend filter ensures we trade with the higher timeframe trend

name = "6h_1d_1w_camarilla_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (based on previous day)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    camarilla_close = np.full(len(df_1d), np.nan)  # previous day close
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        high_prev = df_1d['high'].iloc[i-1]
        low_prev = df_1d['low'].iloc[i-1]
        close_prev = df_1d['close'].iloc[i-1]
        
        camarilla_close[i] = close_prev
        range_prev = high_prev - low_prev
        
        camarilla_r3[i] = close_prev + range_prev * 1.1 / 2
        camarilla_s3[i] = close_prev - range_prev * 1.1 / 2
        camarilla_r4[i] = close_prev + range_prev * 1.1
        camarilla_s4[i] = close_prev - range_prev * 1.1
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = np.full(len(df_1w), np.nan)
    
    if len(close_1w) >= 200:
        multiplier = 2 / (200 + 1)
        ema_200_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * multiplier) + (ema_200_1w[i-1] * (1 - multiplier))
    
    # Align 1d and 1w indicators to 6h timeframe
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_close_6h = align_htf_to_ltf(prices, df_1d, camarilla_close)
    ema_200_1w_6h = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_6h[i]) or 
            np.isnan(camarilla_s3_6h[i]) or 
            np.isnan(camarilla_r4_6h[i]) or 
            np.isnan(camarilla_s4_6h[i]) or 
            np.isnan(camarilla_close_6h[i]) or 
            np.isnan(ema_200_1w_6h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # Trend filter: bullish if price > EMA200_1w, bearish if price < EMA200_1w
        bullish_trend = close[i] > ema_200_1w_6h[i]
        bearish_trend = close[i] < ema_200_1w_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Camarilla S3 OR trend turns bearish
            if close[i] < camarilla_s3_6h[i] or not bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Camarilla R3 OR trend turns bullish
            if close[i] > camarilla_r3_6h[i] or not bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout/mean reversion with volume confirmation and trend filter
            if volume_confirm:
                # Mean reversion long: price crosses above Camarilla S3 in bullish trend
                if close[i] > camarilla_s3_6h[i] and close[i-1] <= camarilla_s3_6h[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Mean reversion short: price crosses below Camarilla R3 in bearish trend
                elif close[i] < camarilla_r3_6h[i] and close[i-1] >= camarilla_r3_6h[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
                # Breakout long: price crosses above Camarilla R4 in bullish trend
                elif close[i] > camarilla_r4_6h[i] and close[i-1] <= camarilla_r4_6h[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price crosses below Camarilla S4 in bearish trend
                elif close[i] < camarilla_s4_6h[i] and close[i-1] >= camarilla_s4_6h[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals