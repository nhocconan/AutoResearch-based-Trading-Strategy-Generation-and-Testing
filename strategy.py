#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channel (20) - breakout levels
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    donchian_high = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily Donchian and ATR to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    atr_14_aligned = align_htf_to_ltf(prices, df_daily, atr_14)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume and volatility
            if (close[i] > donchian_high_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and
                atr_14_aligned[i] > 0.3 * np.mean(atr_14_aligned[max(0, i-50):i+1])):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Donchian low with volume and volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and
                  atr_14_aligned[i] > 0.3 * np.mean(atr_14_aligned[max(0, i-50):i+1])):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the midpoint of the Donchian channel
            donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2.0
            if position == 1:
                if close[i] < donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_DailyDonchian20_Breakout_Volume_Volatility"
timeframe = "4h"
leverage = 1.0