#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout (20) for HTF trend direction and 4h RSI(14) for mean reversion timing.
# In 12h uptrend (price > upper Donchian), wait for 4h RSI < 30 to go long (pullback entry).
# In 12h downtrend (price < lower Donchian), wait for 4h RSI > 70 to go short (bounce entry).
# Volume confirmation ensures momentum validity. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (19-50/year) to minimize fee drag while adapting to trend and mean reversion.

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
    
    # Get 12h and 4h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_4h = get_htf_data(prices, '4h')
    if len(df_12h) < 30 or len(df_4h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian Channel (20) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === 4h Indicators: RSI(14) ===
    close_4h = df_4h['close'].values
    delta = pd.Series(close_4h).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # === 4h Indicators: Volume SMA(20) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 12h uptrend (price > 12h upper Donchian)
        # 2. 4h RSI < 30 (oversold pullback)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i]) and (rsi_4h_aligned[i] < 30) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 12h downtrend (price < 12h lower Donchian)
        # 2. 4h RSI > 70 (overbought bounce)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i]) and (rsi_4h_aligned[i] > 70) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12h_RSI14_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0