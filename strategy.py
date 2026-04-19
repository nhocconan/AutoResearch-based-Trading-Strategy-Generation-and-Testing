#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_Donchian20_Volume_Spike_TrendFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period) from previous 4h bar
    # Upper = highest high over last 20 periods
    high_series = pd.Series(high_4h)
    high_roll_max = high_series.rolling(window=20, min_periods=20).max().values
    # Lower = lowest low over last 20 periods
    low_series = pd.Series(low_4h)
    low_roll_min = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous bar's values (avoid look-ahead)
    donch_high = np.roll(high_roll_max, 1)
    donch_low = np.roll(low_roll_min, 1)
    donch_high[0] = np.nan
    donch_low[0] = np.nan
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 1h timeframe
    donch_high_1h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_1h = align_htf_to_ltf(prices, df_4h, donch_low)
    ema_34_1d_1h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(donch_high_1h[i]) or np.isnan(donch_low_1h[i]) or np.isnan(ema_34_1d_1h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = price > ema_34_1d_1h[i]
        price_below_ema = price < ema_34_1d_1h[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volume spike and above 1d EMA34
            if price > donch_high_1h[i] and volume_spike and price_above_ema:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian low with volume spike and below 1d EMA34
            elif price < donch_low_1h[i] and volume_spike and price_below_ema:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 4h Donchian low (reversal signal)
            if price < donch_low_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price returns above 4h Donchian high (reversal signal)
            if price > donch_high_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals