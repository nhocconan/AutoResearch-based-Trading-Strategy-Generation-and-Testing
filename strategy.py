#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter
# Long when price breaks above Donchian upper band (20) + volume > 1.5x 20-period volume SMA + price > 12h EMA34
# Short when price breaks below Donchian lower band (20) + volume > 1.5x 20-period volume SMA + price < 12h EMA34
# Uses Donchian channels for structure, volume for confirmation, and 12h EMA for trend alignment
# Designed for moderate trade frequency (30-60/year) to balance signal quality and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: EMA34 for Trend Filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === LT Indicators: Donchian Channels (20) ===
    # Calculate Donchian upper/lower bands (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period volume SMA
    vol_series = pd.Series(volume)
    vol_sma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume confirmation
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper band (20)
        # 2. Volume confirmation
        # 3. Price above 12h EMA34 (uptrend filter)
        if (close[i] > donchian_upper[i]) and vol_confirm and (close[i] > ema_34_12h_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower band (20)
        # 2. Volume confirmation
        # 3. Price below 12h EMA34 (downtrend filter)
        elif (close[i] < donchian_lower[i]) and vol_confirm and (close[i] < ema_34_12h_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_12hEMA34_Filter_v1"
timeframe = "4h"
leverage = 1.0