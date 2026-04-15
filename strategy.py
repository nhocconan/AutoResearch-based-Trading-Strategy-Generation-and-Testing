#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume confirmation and 1w trend filter
# Designed for low trade frequency (target 20-40/year) with clear trend following logic
# Williams Alligator uses three smoothed moving averages (Jaw=13, Teeth=8, Lips=5)
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment)
# Requires volume spike and price outside Alligator mouth to avoid false signals
# Works in both bull (trend continuation) and bear (trend continuation) markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Williams Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Williams Alligator: three smoothed moving averages
    # Jaw (13-period): SMMA of median price, smoothed with 8-period
    # Teeth (8-period): SMMA of median price, smoothed with 5-period  
    # Lips (5-period): SMMA of median price, smoothed with 3-period
    median_price = (high_4h + low_4h) / 2
    
    # Smoothed Moving Average (SMMA) - also called Wilder's Moving Average
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV * (period-1) + CURRENT) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    lips = smma(median_price, 5)  # 5-period smoothed
    teeth = smma(median_price, 8) # 8-period smoothed  
    jaw = smma(median_price, 13)  # 13-period smoothed
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (price above/below EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Volume average (20-period on 1d)
    vol_avg = np.full_like(volume_1d, np.nan, dtype=float)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_avg[i] = np.mean(volume_1d[i-19:i+1])
    
    # EMA50 on 1w for trend filter
    ema50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        multiplier = 2 / (50 + 1)
        ema50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema50_1w[i] = (close_1w[i] * multiplier) + (ema50_1w[i-1] * (1 - multiplier))
    
    # ATR for volatility and stoploss (14-period on 4h)
    tr1 = np.maximum(high_4h[1:], low_4h[:-1]) - np.minimum(high_4h[1:], low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        for i in range(13, len(tr)):
            if i == 13:
                atr[i] = np.nanmean(tr[1:15])  # Skip first NaN
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Align all indicators to 4h timeframe
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_aligned[i])):
            continue
        
        # Volatility-adjusted position size (inverse vol)
        vol_factor = np.clip(0.5 * atr_aligned[i] / (close[i] + 1e-10), 0.5, 2.0)
        position_size = base_size / vol_factor
        position_size = np.clip(position_size, 0.15, 0.35)
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: Lips < Teeth < Jaw  
        bearish = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Long entry: bullish alignment + price above teeth + volume spike
        if (bullish and 
            close[i] > teeth_aligned[i] and 
            volume[i] > 1.8 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: bearish alignment + price below teeth + volume spike
        elif (bearish and 
              close[i] < teeth_aligned[i] and 
              volume[i] > 1.8 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: opposite alignment or price crosses jaw
        elif position == 1 and (not bullish or close[i] < jaw_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not bearish or close[i] > jaw_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dVolume_1wEMA_Trend"
timeframe = "4h"
leverage = 1.0