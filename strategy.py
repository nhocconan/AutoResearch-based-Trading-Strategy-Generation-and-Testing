#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and ADX trend filter.
# Long when price > Alligator's Jaw (blue line) AND Teeth > Lips (bullish alignment) AND 1d volume spike AND ADX > 25.
# Short when price < Alligator's Jaw AND Teeth < Lips (bearish alignment) AND 1d volume spike AND ADX > 25.
# Uses Williams Alligator for trend identification, volume for momentum confirmation, and ADX to avoid ranging markets.
# Designed for fewer trades (target: 15-25/year) to reduce fee drag and improve generalization.
# Works in both bull and bear markets by following strong trends with volatility filters.
name = "12h_WilliamsAlligator_Volume_ADX"
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
    
    # Load 1d data for volume spike and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d volume spike: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 2.0
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_values(x, period):
        result = np.zeros_like(x)
        result[period-1] = np.nansum(x[:period])
        for i in range(period, len(x)):
            result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr_14 = smooth_values(tr, 14)
    dm_plus_14 = smooth_values(dm_plus, 14)
    dm_minus_14 = smooth_values(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr_14 > 0, 100 * dm_plus_14 / atr_14, 0)
    di_minus = np.where(atr_14 > 0, 100 * dm_minus_14 / atr_14, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_14 = smooth_values(dx, 14)
    adx_14 = np.where(np.arange(len(adx_14)) < 27, np.nan, adx_14)  # First 27 values invalid
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Load 12h data for Williams Alligator (13,8,5 SMAs of median price)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Median price
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    # Williams Alligator lines
    jaw_12h = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values  # Blue line (13-period)
    teeth_12h = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values    # Red line (8-period)
    lips_12h = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values     # Green line (5-period)
    
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Alligator and ADX calculations
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or np.isnan(lips_12h_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(adx_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw (green > red > blue)
            bullish_alignment = lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i]
            # Bearish alignment: Jaw > Teeth > Lips (blue > red > green)
            bearish_alignment = jaw_12h_aligned[i] > teeth_12h_aligned[i] > lips_12h_aligned[i]
            
            # Long condition: price > Jaw, bullish alignment, volume spike, strong trend (ADX > 25)
            long_condition = (close[i] > jaw_12h_aligned[i]) and bullish_alignment and vol_spike_1d_aligned[i] and (adx_14_aligned[i] > 25)
            # Short condition: price < Jaw, bearish alignment, volume spike, strong trend (ADX > 25)
            short_condition = (close[i] < jaw_12h_aligned[i]) and bearish_alignment and vol_spike_1d_aligned[i] and (adx_14_aligned[i] > 25)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below Jaw or alignment turns bearish or ADX weakens (< 20)
            if (close[i] < jaw_12h_aligned[i]) or (teeth_12h_aligned[i] < lips_12h_aligned[i]) or (adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above Jaw or alignment turns bullish or ADX weakens (< 20)
            if (close[i] > jaw_12h_aligned[i]) or (teeth_12h_aligned[i] > lips_12h_aligned[i]) or (adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals