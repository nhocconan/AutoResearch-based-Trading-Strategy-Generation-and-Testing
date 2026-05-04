#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power combo with 1d trend filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via alignment
# Elder Ray Power (Bull/Bear) measures trend strength relative to EMA13
# 1d EMA34 trend filter ensures alignment with higher timeframe direction
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in bull markets (alligator bullish alignment + positive Bull Power) and bear markets (alligator bearish alignment + negative Bear Power)
# Uses volume confirmation to avoid whipsaws in low participation environments

name = "6h_WilliamsAlligator_ElderRay_1dEMA34_Volume"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate Williams Alligator on 6h data
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    close_series = pd.Series(close)
    jaw = close_series.ewm(alpha=1/13, adjust=False, min_periods=13).mean().values  # approximates SMMA(13)
    teeth = close_series.ewm(alpha=1/8, adjust=False, min_periods=8).mean().values   # approximates SMMA(8)
    lips = close_series.ewm(alpha=1/5, adjust=False, min_periods=5).mean().values    # approximates SMMA(5)
    
    # Calculate Elder Ray Power (requires EMA13)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup period for Alligator
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish alignment (Lips > Teeth > Jaw) AND positive Bull Power AND 1d uptrend AND volume spike
            if (lips[i] > teeth[i] > jaw[i] and 
                bull_power[i] > 0 and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator bearish alignment (Jaw > Teeth > Lips) AND negative Bear Power AND 1d downtrend AND volume spike
            elif (jaw[i] > teeth[i] > lips[i] and 
                  bear_power[i] < 0 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power becomes negative OR 1d trend turns down
            if (jaw[i] >= teeth[i] or  # Jaw crosses above Teeth (death cross)
                bull_power[i] <= 0 or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power becomes positive OR 1d trend turns up
            if (lips[i] >= teeth[i] or  # Lips crosses above Teeth (golden cross)
                bear_power[i] >= 0 or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals