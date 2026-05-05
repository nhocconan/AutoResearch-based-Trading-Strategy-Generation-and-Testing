#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla H3/L3 fade with 1d volume spike and 1w EMA50 trend filter
# Long when price breaks below weekly Camarilla L3 AND 1d volume > 2.0 * avg_volume(20) AND 1w close > 1w EMA50 (uptrend)
# Short when price breaks above weekly Camarilla H3 AND 1d volume > 2.0 * avg_volume(20) AND 1w close < 1w EMA50 (downtrend)
# Exit when price crosses back through the weekly Camarilla midpoint (H3/L3 average)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 60-120 total trades over 4 years (15-30/year) for 6h timeframe
# Weekly Camarilla H3/L3 levels provide fade opportunities in ranging markets
# 1d volume confirmation validates breakout strength
# 1w EMA50 filter ensures we fade only in the direction of the weekly trend (avoid fading strong trends)

name = "6h_WeeklyCamarillaH3L3_Fade_1dVolumeSpike_1wEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Camarilla calculation and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (H3, L3, midpoint)
    # Camarilla: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    high_low_1w = high_1w - low_1w
    camarilla_h3_1w = close_1w + 1.1 * high_low_1w * 1.1 / 4.0
    camarilla_l3_1w = close_1w - 1.1 * high_low_1w * 1.1 / 4.0
    camarilla_mid_1w = (camarilla_h3_1w + camarilla_l3_1w) / 2.0
    
    # Align weekly Camarilla to 6h timeframe (wait for completed weekly bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1w, camarilla_mid_1w)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for volume average
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks below weekly Camarilla L3, 1d volume spike, 1w close > 1w EMA50 (uptrend)
            if (close[i] < camarilla_l3_aligned[i] and 
                volume_spike_1d_aligned[i] and 
                close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False):  # Use latest weekly close for trend
                signals[i] = 0.25
                position = 1
            # Short: price breaks above weekly Camarilla H3, 1d volume spike, 1w close < 1w EMA50 (downtrend)
            elif (close[i] > camarilla_h3_aligned[i] and 
                  volume_spike_1d_aligned[i] and 
                  close_1w[-1] < ema_50_1w[-1] if len(close_1w) > 0 else False):  # Use latest weekly close for trend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back above weekly Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back below weekly Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals