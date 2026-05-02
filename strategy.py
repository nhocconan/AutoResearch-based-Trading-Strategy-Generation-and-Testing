#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide precise intraday support/resistance; breakout above R3 or below S3 signals strong momentum
# 1d EMA34 ensures alignment with daily trend (price > EMA for longs, < EMA for shorts)
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Uses discrete position sizing 0.25 to minimize fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by requiring volume confirmation and trend alignment

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # We use R3 and S3 levels (most significant for breakouts)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    camarilla_r3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.25 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe (previous day's levels available at 12h open)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla, EMA and volume MA)
    start_idx = 50  # max(20 for volume, 34 for EMA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + price > 1d EMA + volume confirmation
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + price < 1d EMA + volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to Camarilla H3 level (3/8 level)
            camarilla_h3 = prev_close + 0.25 * (prev_high - prev_low)  # H3 = close + 0.25*(high-low)
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            if close[i] < camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises to Camarilla L3 level (5/8 level)
            camarilla_l3 = prev_close - 0.25 * (prev_high - prev_low)  # L3 = close - 0.25*(high-low)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            if close[i] > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals