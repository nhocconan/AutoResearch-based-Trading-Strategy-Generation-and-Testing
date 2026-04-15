#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA34 trend filter
# Long when price breaks above Donchian upper (20) + volume > 1.5x 20-period volume SMA + price > 1d EMA34
# Short when price breaks below Donchian lower (20) + volume > 1.5x 20-period volume SMA + price < 1d EMA34
# Uses Donchian channels for structure, volume for confirmation, and 1d EMA for trend alignment
# Designed for low trade frequency (20-50/year) to minimize fee drag while capturing breakouts with trend

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA34 for Trend Filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === LT Indicators: Donchian Channels (20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation
    vol_series = pd.Series(volume)
    vol_sma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (20)
        # 2. Volume confirmation
        # 3. Price above 1d EMA34 (uptrend filter)
        if (close[i] > donchian_upper[i]) and vol_confirm and (close[i] > ema_34_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (20)
        # 2. Volume confirmation
        # 3. Price below 1d EMA34 (downtrend filter)
        elif (close[i] < donchian_lower[i]) and vol_confirm and (close[i] < ema_34_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_1dEMA34_Filter_v1"
timeframe = "4h"
leverage = 1.0