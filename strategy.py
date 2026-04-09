#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h volume confirmation + ADX regime filter
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Strong Bull Power + rising ADX = bullish momentum; Strong Bear Power + rising ADX = bearish momentum
# 12h volume spike confirms institutional participation
# ADX > 25 filters for trending markets, ADX < 20 for ranging (avoid chop)
# Works in bull/bear: ADX regime adapts, Elder Ray captures momentum with volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_12h_elder_ray_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for volume, ADX, and EMA13 calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA13 for Elder Ray
    close_12h = df_12h['close'].values
    ema13_12h = pd.Series(close_12h).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate 12h Bull Power and Bear Power
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 12h average volume (20-period)
    volume_12h = df_12h['volume'].values
    avg_volume_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe (wait for 12h bar close)
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    # Calculate 6h EMA13 for entry timing
    ema13_6h = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_12h_aligned[i]) or np.isnan(bear_power_12h_aligned[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(ema13_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x 12h average volume
        volume_confirmed = volume[i] > 1.8 * avg_volume_12h_aligned[i]
        
        # Regime filter: ADX > 25 = trending (follow momentum), ADX < 20 = ranging (avoid)
        trending_regime = adx_12h_aligned[i] > 25
        ranging_regime = adx_12h_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes strongly negative OR regime shifts to ranging
            if bear_power_12h_aligned[i] < -0.5 * np.std(bear_power_12h_aligned[max(0, i-50):i+1]) or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power becomes strongly positive OR regime shifts to ranging
            if bull_power_12h_aligned[i] > 0.5 * np.std(bull_power_12h_aligned[max(0, i-50):i+1]) or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if trending_regime and volume_confirmed:
                # Follow momentum in trending regime with volume confirmation
                if bull_power_12h_aligned[i] > 0.3 * np.std(bull_power_12h_aligned[max(0, i-50):i+1]) and close[i] > ema13_6h[i]:
                    position = 1
                    signals[i] = 0.25
                elif bear_power_12h_aligned[i] < -0.3 * np.std(bear_power_12h_aligned[max(0, i-50):i+1]) and close[i] < ema13_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals