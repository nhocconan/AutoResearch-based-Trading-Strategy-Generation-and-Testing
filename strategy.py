#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Weekly Trend Filter
# Williams Alligator (JAW=TEETH=LIPS smoothed SMAs) identifies trending vs ranging markets.
# Long when LIPS > TEETH > JAW (bullish alignment) with volume spike > 2x 24-bar median.
# Short when LIPS < TEETH < JAW (bearish alignment) with volume spike.
# Weekly trend filter: only take longs when price > weekly EMA20, shorts when price < weekly EMA20.
# Designed to work in trending markets (both bull and bear) while avoiding chop.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 12h close
    def smma(arr, period):
        smoothed = np.full_like(arr, np.nan)
        smoothed[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
        return smoothed
    
    jaw = smma(close, 13)  # 13-period SMMA smoothed 8 bars ahead
    teeth = smma(close, 8)  # 8-period SMMA smoothed 5 bars ahead
    lips = smma(close, 5)   # 5-period SMMA smoothed 3 bars ahead
    
    # Shift jaws/teeth/lips to align with Alligator logic (future shift)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # 1-day volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_24 = pd.Series(vol_1d).rolling(window=24, min_periods=1).mean()
    vol_ma_24_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_24.values)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean()
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w.values)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma_24_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volume spike: current 12h volume > 2x daily average volume (scaled)
        # Approximate: 12h volume > 0.5 * daily volume * 2 = daily volume
        volume_spike = volume[i] > vol_ma_24_aligned[i]
        
        # Weekly trend filter
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        # Long: Bullish alignment + volume spike + price above weekly EMA
        if bullish and volume_spike and price_above_weekly_ema:
            signals[i] = 0.25
        
        # Short: Bearish alignment + volume spike + price below weekly EMA
        elif bearish and volume_spike and price_below_weekly_ema:
            signals[i] = -0.25
        
        # Exit: Alligator lines intertwine (no clear alignment) or volume drops
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and not (bullish and volume_spike)) or
               (signals[i-1] == -0.25 and not (bearish and volume_spike)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_1wTrend"
timeframe = "12h"
leverage = 1.0