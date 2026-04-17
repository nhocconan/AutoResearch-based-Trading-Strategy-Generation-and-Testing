#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 4h timeframe
    high_20_4h = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_4h = align_htf_to_ltf(prices, df_1d, low_20)
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume filter: current volume > 1.5 * 50-period average
    volume_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need daily Donchian(20), ATR(14), volume MA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_4h[i]) or 
            np.isnan(low_20_4h[i]) or 
            np.isnan(atr_14_4h[i]) or 
            np.isnan(volume_ma50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma50[i])
        
        # Volatility filter: current ATR > 0.8 * 20-period ATR average (avoid low volatility chop)
        atr_ma20 = pd.Series(atr_14_4h).rolling(window=20, min_periods=20).mean()
        atr_ma20_val = atr_ma20.iloc[i] if not np.isnan(atr_ma20.iloc[i]) else atr_14_4h[i]
        volatility_filter = atr_14_4h[i] > (0.8 * atr_ma20_val)
        
        # Price relative to daily Donchian channels
        price_above_high20 = close[i] > high_20_4h[i]
        price_below_low20 = close[i] < low_20_4h[i]
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume and volatility
            if (price_above_high20 and volume_filter and volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Donchian low with volume and volatility
            elif (price_below_low20 and volume_filter and volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily Donchian low OR ATR drops below threshold
            if (price_below_low20) or (atr_14_4h[i] < (0.5 * atr_ma20_val)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily Donchian high OR ATR drops below threshold
            if (price_above_high20) or (atr_14_4h[i] < (0.5 * atr_ma20_val)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyDonchian_Breakout_VolVol"
timeframe = "4h"
leverage = 1.0