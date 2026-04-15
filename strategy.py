#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for direction and 1d volume spike for confirmation.
# Uses 4h for signal direction (reducing trade frequency) and 1h only for entry timing precision.
# Volume spike filter avoids low-quality breakouts. Session filter (08-20 UTC) reduces noise.
# Designed to work in both bull (breakouts continue) and bear (failed breaks reverse) markets.
# Target: 15-35 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h
    upper_20_1h = align_htf_to_ltf(prices, df_4h, upper_20_4h)
    lower_20_1h = align_htf_to_ltf(prices, df_4h, lower_20_4h)
    
    # Get 1d HTF data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average
    vol_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20_1h = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    # Calculate 1h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_1h[i]) or np.isnan(lower_20_1h[i]) or 
            np.isnan(vol_20_1h[i]) or np.isnan(atr_14[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike: current 1h volume > 2.0 x 1d average volume (scaled to 1h)
        # 1d average volume / 24 = approximate 1h average volume
        vol_spike = volume[i] > (2.0 * vol_20_1h[i] / 24.0)
        
        # Long conditions:
        # 1. 1h price breaks above 4h Donchian upper (20) - bullish breakout
        # 2. Volume confirmation: volume spike
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_1h[i] and
            vol_spike and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 1h price breaks below 4h Donchian lower (20) - bearish breakdown
        # 2. Volume confirmation: volume spike
        # 3. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_1h[i] and
              vol_spike and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_Donchian20_1d_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0