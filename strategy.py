#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above R3 in 12h bullish trend with volume spike.
Short when price breaks below S3 in 12h bearish trend with volume spike.
Camarilla levels provide precise intraday support/resistance. 12h EMA50 filters counter-trend trades.
Volume spike confirms institutional participation. Discrete sizing (0.25) minimizes fee churn.
Targets 20-50 trades/year on 4h timeframe. Works in bull/bear by following 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h trend: price above/below EMA50
    trend_bullish = close_12h > ema_50_12h
    trend_bearish = close_12h < ema_50_12h
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish.astype(float))
    
    # Get 1d data for Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We only need R3 and S3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day has no previous - set to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R3 and S3
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA, 1 for Camarilla)
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with 12h bullish trend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_bullish_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with 12h bearish trend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_bearish_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Camarilla S3 OR 12h trend turns bearish
            if (close[i] < camarilla_s3_aligned[i] or not trend_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Camarilla R3 OR 12h trend turns bullish
            if (close[i] > camarilla_r3_aligned[i] or not trend_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0