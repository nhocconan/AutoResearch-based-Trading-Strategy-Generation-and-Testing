#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R extremes for mean reversion in ranging markets
# combined with 1d EMA200 trend filter and volume confirmation. In ranging markets (price between
# Williams %R oversold/overbought levels), fade extremes; in trending markets (price outside
# extreme Williams %R levels), continuation. Volume filter ensures momentum validity.
# Designed for low trade frequency (7-25/year) to minimize fee drag while adapting to regime.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1w and 1d HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w Indicators: Williams %R (14-period) ===
    highest_high_1w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_1w - df_1w['close'].values) / (highest_high_1w - lowest_low_1w)
    williams_r_1w = np.where((highest_high_1w - lowest_low_1w) == 0, -50, williams_r_1w)  # avoid div by zero
    
    # Align to 1d timeframe
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(200) for trend bias
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_1w_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION BASED ON WILLIAMS %R ===
        # Ranging market: Williams %R between -80 and -20 (not extreme)
        # Oversold: Williams %R <= -80 (potential long)
        # Overbought: Williams %R >= -20 (potential short)
        # Extreme trending: Williams %R < -90 or > -10 (strong momentum)
        
        wr = williams_r_1w_aligned[i]
        in_range = (-80 < wr < -20)
        oversold = wr <= -80
        overbought = wr >= -20
        extreme_bear = wr < -90
        extreme_bull = wr > -10
        
        # === LONG CONDITIONS ===
        # 1. In ranging market AND oversold (mean reversion long)
        # 2. OR in extreme bear AND price > EMA200 (potential reversal long)
        # 3. Volume confirmation
        if vol_confirm:
            if (in_range and oversold) or \
               (extreme_bear and close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In ranging market AND overbought (mean reversion short)
        # 2. OR in extreme bull AND price < EMA200 (potential reversal short)
        # 3. Volume confirmation
        elif vol_confirm:
            if (in_range and overbought) or \
               (extreme_bull and close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_WilliamsR_EMA200_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0