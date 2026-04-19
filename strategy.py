#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour volume confirmation and ATR-based volatility filter.
# Long when price breaks above 4h Donchian upper band (20-period) with 12h volume > 1.8x average and price > 12h EMA50 (uptrend).
# Short when price breaks below 4h Donchian lower band (20-period) with 12h volume > 1.8x average and price < 12h EMA50 (downtrend).
# Exit when price returns to the 4h Donchian middle band or reverses to opposite band.
# Designed for 20-40 trades/year per symbol with clear trend-following logic and volatility filtering to reduce false breakouts.

name = "4h_Donchian20_Breakout_Volume_EMA50Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for volume confirmation and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    close_12h = df_12h['close'].values
    
    # 4h Donchian channel (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_max_20 + low_min_20) / 2
    
    # 12h volume average (20-period) for confirmation
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        
        # Get current levels
        upper_band = high_max_20[i]
        lower_band = low_min_20[i]
        middle_band = donchian_middle[i]
        ema_50 = ema_50_12h_aligned[i]
        
        if position == 0:
            # Long breakout: price > upper band with volume confirmation and uptrend
            if price > upper_band and vol > 1.8 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band with volume confirmation and downtrend
            elif price < lower_band and vol > 1.8 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band or breaks below lower band (reversal)
            if price <= middle_band or price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band or breaks above upper band (reversal)
            if price >= middle_band or price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals