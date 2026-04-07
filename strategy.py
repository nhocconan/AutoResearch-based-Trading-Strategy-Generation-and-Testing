#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Ehlers Fisher Transform + Weekly Trend Filter + Volume Spike
# Hypothesis: Fisher Transform identifies extreme price reversals with low lag.
# Weekly trend filter ensures alignment with higher-timeframe momentum.
# Volume spike confirms institutional participation in the reversal.
# Designed for 6h timeframe with low trade frequency (12-37/year).
# Works in bull via Fisher long signals + weekly uptrend + volume, 
# in bear via Fisher short signals + weekly downtrend + volume.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_fisher_transform_1w_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Ehlers Fisher Transform (9-period)
    # Step 1: Normalize price to [-1, 1] range over lookback period
    def normalize_price(high, low, lookback=9):
        nn = np.zeros_like(high)
        hh = np.zeros_like(high)
        for i in range(len(high)):
            if i < lookback - 1:
                nn[i] = low[:i+1].min()
                hh[i] = high[:i+1].max()
            else:
                nn[i] = low[i-lookback+1:i+1].min()
                hh[i] = high[i-lookback+1:i+1].max()
        # Avoid division by zero
        diff = hh - nn
        diff[diff == 0] = 1e-10
        return 2 * ((high - nn) / diff - 0.5)
    
    # Step 2: Apply Gaussian smoothing
    def gaussian_smooth(values, alpha=0.33):
        result = np.zeros_like(values)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # Step 3: Fisher Transform
    price_norm = normalize_price(high, low, 9)
    smoothed = gaussian_smooth(price_norm, 0.33)
    # Clamp to prevent math domain error
    smoothed = np.clip(smoothed, -0.999, 0.999)
    fisher = 0.5 * np.log((1 + smoothed) / (1 - smoothed))
    
    # Weekly trend filter: EMA(20) of weekly close
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(fisher[i]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: Fisher crosses below zero OR weekly trend turns bearish
            if fisher[i] < 0 or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # Short position
            # Exit: Fisher crosses above zero OR weekly trend turns bullish
            if fisher[i] > 0 or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.28
        else:  # Flat, look for entry
            if vol_ok:
                # Long: Fisher crosses above -0.5 (oversold bounce) with weekly uptrend
                if fisher[i] > -0.5 and (i == 20 or fisher[i-1] <= -0.5) and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.28
                # Short: Fisher crosses below 0.5 (overbought reversal) with weekly downtrend
                elif fisher[i] < 0.5 and (i == 20 or fisher[i-1] >= 0.5) and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.28
    
    return signals