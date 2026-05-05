#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Long when: Jaw < Teeth < Lips (bullish alignment), price > Lips, volume > 1.5x 20-bar average
# Short when: Jaw > Teeth > Lips (bearish alignment), price < Lips, volume > 1.5x 20-bar average
# Exit when: Alligator alignment reverses (Teeth crosses Jaw or Lips)
# Uses Williams Alligator from 1d for trend structure, effective in both bull (continuation) and bear (reversal via alignment) markets.
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_WilliamsAlligator_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) SMAs of median price
    # Median price = (high + low) / 2
    median_price_1d = (high_1d + low_1d) / 2
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 1d timeframe (already 1d, but using align for consistency and proper Bar close timing)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Jaw < Teeth < Lips
            bullish_align = (jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i])
            # Bearish alignment: Jaw > Teeth > Lips
            bearish_align = (jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i])
            
            # Long conditions: bullish alignment, price > Lips, volume filter, and above 1w EMA50
            if (bullish_align and 
                close[i] > lips_1d_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment, price < Lips, volume filter, and below 1w EMA50
            elif (bearish_align and 
                  close[i] < lips_1d_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bullish alignment breaks (Teeth crosses below Jaw or Lips crosses below Teeth)
            if (teeth_1d_aligned[i] < jaw_1d_aligned[i]) or (lips_1d_aligned[i] < teeth_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bearish alignment breaks (Teeth crosses above Jaw or Lips crosses above Teeth)
            if (teeth_1d_aligned[i] > jaw_1d_aligned[i]) or (lips_1d_aligned[i] > teeth_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals