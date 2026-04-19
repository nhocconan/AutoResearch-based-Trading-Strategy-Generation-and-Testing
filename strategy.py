#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day EMA trend filter and volume confirmation.
# Long when: Close breaks above Donchian high (20-period) AND price > EMA50(1d) AND volume > 1.5x 20-period average
# Short when: Close breaks below Donchian low (20-period) AND price < EMA50(1d) AND volume > 1.5x 20-period average
# Exit: Opposite Donchian breakout (long exits on lower band break, short exits on upper band break)
# Donchian channels capture breakouts, EMA50 filters trend direction, volume confirms conviction.
# Target: 20-40 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).

name = "4h_Donchian_EMA50_Volume_Breakout"
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
    
    # 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        ema50 = ema50_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Close breaks above upper band, price > EMA50, volume confirmation
            if (price > upper_band and price > ema50 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Close breaks below lower band, price < EMA50, volume confirmation
            elif (price < lower_band and price < ema50 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close breaks below lower band (opposite Donchian)
            if price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close breaks above upper band (opposite Donchian)
            if price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals