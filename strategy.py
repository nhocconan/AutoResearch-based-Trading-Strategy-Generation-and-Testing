#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Camarilla R3/S3 levels from 1d provide institutional support/resistance for mean reversion
# 12h EMA50 provides intermediate trend filter to avoid counter-trend breakouts
# Volume spike (>1.8 * 20-period EMA on 6h) confirms strong participation
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure
# Works in bull (continuation via trend filter) and bear (mean reversion at R3/S3) markets
# Designed to avoid overtrading by requiring confluence of price level, trend, and volume

name = "6h_Camarilla_R3S3_12hEMA50_Trend_Volume"
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
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical Camarilla formula: Pivot = (H+L+C)/3
    # R3 = Pivot + (H-L)*1.1/2, S3 = Pivot - (H-L)*1.1/2
    # R4 = Pivot + (H-L)*1.1, S4 = Pivot - (H-L)*1.1
    # We use R3/S3 for mean reversion entries, R4/S4 for breakout confirmation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot_1d = typical_price.values
    hl_range = (df_1d['high'] - df_1d['low']).values
    
    r3_1d = pivot_1d + (hl_range * 1.1 / 2)
    s3_1d = pivot_1d - (hl_range * 1.1 / 2)
    r4_1d = pivot_1d + (hl_range * 1.1)
    s4_1d = pivot_1d - (hl_range * 1.1)
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar close)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.8 * 20-period EMA (6h)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50
        bullish_bias = close[i] > ema_50_12h_aligned[i]
        bearish_bias = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above R3 with volume spike (continuation)
                # or price bounces off S3 with volume spike (mean reversion)
                if ((close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1]) or
                    (close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1])) and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below S3 with volume spike (continuation)
                # or price bounces off R3 with volume spike (mean reversion)
                if ((close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1]) or
                    (close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1])) and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around 12h EMA50
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 or above R4 (take profit)
            if close[i] < s3_1d_aligned[i] or close[i] > r4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or below S4 (take profit)
            if close[i] > r3_1d_aligned[i] or close[i] < s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals