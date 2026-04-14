# 1. Hypothesis: 1d price breaking above/below 4-hour Donchian(20) channel with volume above 1.5x 4-period average and 4-hour ADX > 25.
# Trades in direction of 4-hour trend to avoid counter-trend whipsaws. Uses Donchian for clear breakout signals and ADX for trend strength.
# Targets 12-37 trades/year per symbol (48-148 total over 4 years).
# Uses 4h timeframe with 1h/4h multi-timeframe for trend confirmation.

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
    
    # Calculate 4-hour Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper and lower bands
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4-hour ADX (14-period)
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - low_4h[:-1])  # high - prev close
    tr3 = np.abs(low_4h[1:] - low_4h[:-1])  # low - prev close
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_4h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4-period average volume (4h periods in a day)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned indicators
        upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)[i]
        lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)[i]
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)[i]
        vol_ma_4_val = vol_ma_4[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(upper_20_aligned) or np.isnan(lower_20_aligned) or 
            np.isnan(adx_4h_aligned) or np.isnan(vol_ma_4_val)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma_4_val
        
        # ADX trend filter (> 25)
        trend_filter = adx_4h_aligned > 25
        
        if position == 0:  # No position - look for entries
            if volume_confirm and trend_filter:
                # Long: price breaks above upper Donchian band
                if close[i] > upper_20_aligned and close[i-1] <= upper_20_aligned:
                    position = 1
                    signals[i] = position_size
                # Short: price breaks below lower Donchian band
                elif close[i] < lower_20_aligned and close[i-1] >= lower_20_aligned:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below lower band
            if close[i] < lower_20_aligned and close[i-1] >= lower_20_aligned:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above upper band
            if close[i] > upper_20_aligned and close[i-1] <= upper_20_aligned:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_4hDonchian20_4hADX25_Volume1.5x_v1"
timeframe = "4h"
leverage = 1.0