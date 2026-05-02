#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3, S3) from 1w data for institutional breakout levels
# 1w EMA34 filter ensures alignment with long-term trend to avoid counter-trend whipsaws
# Volume spike (>2.0 * 20-period EMA) confirms institutional participation
# Designed for low trade frequency: ~7-25 trades/year per symbol with 0.25 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# BTC/ETH focused: requires 1w trend alignment and volume spike to avoid SOL-only bias

name = "1d_Camarilla_R3S3_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for Camarilla pivot calculation (using previous week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous 1w bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    camarilla_r3 = close_1w + (range_1w * 1.1 / 2.0)
    camarilla_s3 = close_1w - (range_1w * 1.1 / 2.0)
    
    # Align Camarilla levels to 1d timeframe (use previous week's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # 1w HTF data for EMA34 trend filter
    df_1w_trend = get_htf_data(prices, '1w')
    if len(df_1w_trend) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w_trend = df_1w_trend['close'].values
    ema_34_1w = pd.Series(close_1w_trend).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w_trend, ema_34_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (strict filter)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA34
        bullish_bias = close[i] > ema_34_1w_aligned[i]
        bearish_bias = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Camarilla R3 with volume spike
                if close[i] > camarilla_r3_aligned[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Camarilla S3 with volume spike
                if close[i] < camarilla_s3_aligned[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA34
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 or price below 1w EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 or price above 1w EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals