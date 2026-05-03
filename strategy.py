#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide high-probability reversal/breakout zones derived from prior day's range.
# R3 (resistance 3) and S3 (support 3) are strong breakout levels; breaks often continue with momentum.
# 1d EMA34 ensures we trade with the daily trend (longs in uptrend, shorts in downtrend).
# Volume spike confirms institutional participation and reduces false breakouts.
# Designed for 20-40 trades/year on 4h to minimize fee drag and improve test generalization.
# Works in bull markets via trend continuation and in bear markets via shorting breakdowns.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior day's Camarilla levels (R3, S3)
    # Use prior day's high, low, close to avoid look-ahead
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 formulas
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (already delayed by shift(1) for prior day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: bullish breakout above R3 in 1d uptrend with volume spike
            if breakout_up and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown below S3 in 1d downtrend with volume spike
            elif breakout_down and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla H3/L3 level or loses 1d uptrend
            camarilla_h3 = prior_close[i-1] + (prior_high[i-1] - prior_low[i-1]) * 1.1 / 6 if i >= 1 else camarilla_r3_aligned[i]
            camarilla_l3 = prior_close[i-1] - (prior_high[i-1] - prior_low[i-1]) * 1.1 / 6 if i >= 1 else camarilla_s3_aligned[i]
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3) if i >= 1 else camarilla_r3_aligned[i]
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3) if i >= 1 else camarilla_s3_aligned[i]
            
            if close[i] < camarilla_h3_aligned[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla H3/L3 level or loses 1d downtrend
            camarilla_h3 = prior_close[i-1] + (prior_high[i-1] - prior_low[i-1]) * 1.1 / 6 if i >= 1 else camarilla_r3_aligned[i]
            camarilla_l3 = prior_close[i-1] - (prior_high[i-1] - prior_low[i-1]) * 1.1 / 6 if i >= 1 else camarilla_s3_aligned[i]
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3) if i >= 1 else camarilla_r3_aligned[i]
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3) if i >= 1 else camarilla_s3_aligned[i]
            
            if close[i] > camarilla_l3_aligned[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals