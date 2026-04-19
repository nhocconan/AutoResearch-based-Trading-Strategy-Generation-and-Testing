#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Long when: price breaks above 4h Donchian upper band (20-period) + volume > 1.5x 20-period average + price > 12h EMA34
# Short when: price breaks below 4h Donchian lower band (20-period) + volume > 1.5x 20-period average + price < 12h EMA34
# Exit when price returns to 4h Donchian middle band or reverses to opposite band.
# Uses Donchian channels for breakout detection, volume to confirm momentum, and higher timeframe trend to avoid counter-trend trades.
# Designed for ~20-40 trades/year per symbol to avoid fee drag.
name = "4h_Donchian20_Breakout_Trend_Volume"
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
    
    # 4h Donchian channels (20-period)
    donchian_len = 20
    upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    middle = (upper + lower) / 2
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, donchian_len)  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Get current levels
        up = upper[i]
        low_band = lower[i]
        mid = middle[i]
        ema_34 = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long breakout: price > upper band with volume confirmation and uptrend
            if price > up and vol > 1.5 * vol_ma and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band with volume confirmation and downtrend
            elif price < low_band and vol > 1.5 * vol_ma and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band or breaks below lower band (reversal)
            if price <= mid or price < low_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band or breaks above upper band (reversal)
            if price >= mid or price > up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals