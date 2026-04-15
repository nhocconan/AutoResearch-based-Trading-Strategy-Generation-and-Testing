#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Donchian breakout with volume confirmation and 1d EMA trend filter
# Long when price breaks above 4h Donchian upper (20) + volume > 2.0x 20-period volume avg + price > 1d EMA34
# Short when price breaks below 4h Donchian lower (20) + volume > 2.0x 20-period volume avg + price < 1d EMA34
# Uses 4h Donchian channels for structure and 1d EMA34 for trend alignment to reduce false breakouts
# Designed for low trade frequency (15-37/year) on 1h timeframe with session filter (08-20 UTC) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channels (20) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian calculations
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # === 1d Indicators: EMA34 for Trend Filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20)
        # 2. Volume confirmation
        # 3. Price above 1d EMA34 (uptrend filter)
        if (close[i] > donchian_high_20_aligned[i]) and vol_confirm and (close[i] > ema_34_1d_aligned[i]):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20)
        # 2. Volume confirmation
        # 3. Price below 1d EMA34 (downtrend filter)
        elif (close[i] < donchian_low_20_aligned[i]) and vol_confirm and (close[i] < ema_34_1d_aligned[i]):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_4h_Donchian20_Volume_1dEMA34_Filter_v1"
timeframe = "1h"
leverage = 1.0