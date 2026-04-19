#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h ADX trend filter.
# Long when price breaks above upper band AND ADX > 25 (trending) AND volume spike (>1.5x average).
# Short when price breaks below lower band AND ADX > 25 AND volume spike.
# Uses 12h ADX to filter for trending conditions, avoiding whipsaw in sideways markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_Donchian20_12hADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX calculation and Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 12h data
    # Upper band = max(high, 20), Lower band = min(low, 20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe (wait for 12h bar close)
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # Calculate ADX (14-period) on 12h data
    # ADX calculation requires +DI, -DI, and DX
    tr1 = high_12h[1:] - low_12h[:-1]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed averages
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 4h timeframe (wait for 12h bar close)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need Donchian and ADX data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_val > 25
        
        if position == 0:
            # Enter long: price breaks above upper band AND trending AND volume spike
            if price > upper and trending and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band AND trending AND volume spike
            elif price < lower and trending and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower band or trend weakens
            if price < lower or adx_val < 20:  # Exit if trend weakens (ADX < 20)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above upper band or trend weakens
            if price > upper or adx_val < 20:  # Exit if trend weakens (ADX < 20)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals