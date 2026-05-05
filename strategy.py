#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator Jaw breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above Alligator Jaw (13-period SMMA of median price) AND price > EMA50(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Alligator Jaw AND price < EMA50(1d) AND volume > 2.0x 20-period average
# Exit when price returns to Alligator Jaw (mean reversion) OR trend flips (price crosses EMA50(1d))
# Williams Alligator is a trend-following system using smoothed moving averages (SMMA) of median price
# Jaw (blue) = 13-period SMMA of median price, Teeth (red) = 8-period, Lips (green) = 5-period
# We use Jaw as the primary trend filter as it's the slowest and most reliable
# 1d EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws
# Volume spike confirms institutional participation
# Target: 12-37 trades/year per symbol (50-150 total over 4 years)
# Discrete sizing (0.25) to limit fee drag

name = "6h_WilliamsAlligator_JawBreak_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator Jaw on 6h timeframe
    # Jaw = 13-period SMMA of median price ((high + low) / 2)
    median_price = (high + low) / 2
    jaw = np.full(n, np.nan)
    
    if len(median_price) >= 13:
        # Calculate SMMA (Smoothed Moving Average) for Jaw
        # SMMA is similar to EMA but with different smoothing factor
        # SMMA(t) = (SMMA(t-1) * (period-1) + price(t)) / period
        # We'll calculate it manually since pandas doesn't have built-in SMMA
        smma = np.full(n, np.nan)
        smma[12] = np.mean(median_price[0:13])  # First value is simple average
        
        for i in range(13, n):
            if not np.isnan(smma[i-1]) and not np.isnan(median_price[i]):
                smma[i] = (smma[i-1] * 12 + median_price[i]) / 13
        
        jaw = smma
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Alligator Jaw AND price > EMA50(1d) AND volume spike
            if (close[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Alligator Jaw AND price < EMA50(1d) AND volume spike
            elif (close[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Alligator Jaw (mean reversion) OR price < EMA50(1d) (trend flip)
            if (close[i] <= jaw[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Alligator Jaw (mean reversion) OR price > EMA50(1d) (trend flip)
            if (close[i] >= jaw[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals