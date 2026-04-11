#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed 1d bars
    donchian_high_1d = np.roll(donchian_high_1d, 1)
    donchian_low_1d = np.roll(donchian_low_1d, 1)
    donchian_high_1d[0] = np.nan
    donchian_low_1d[0] = np.nan
    
    # Align 1d Donchian levels to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # 4h momentum: price > 20-period SMA (trend filter)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or
            np.isnan(sma_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: strong threshold
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price above/below 20-period SMA
        price_above_sma = price_close > sma_20[i]
        price_below_sma = price_close < sma_20[i]
        
        # Long conditions: price breaks above 1d Donchian high with volume and uptrend
        long_signal = volume_confirmed and price_above_sma and (price_high > donchian_high_4h[i])
        
        # Short conditions: price breaks below 1d Donchian low with volume and downtrend
        short_signal = volume_confirmed and price_below_sma and (price_low < donchian_low_4h[i])
        
        # Exit when price returns to the opposite Donchian level (mean reversion)
        exit_long = position == 1 and price_close < donchian_low_4h[i]
        exit_short = position == -1 and price_close > donchian_high_4h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1d Donchian breakouts with volume confirmation and trend filter capture
# strong momentum moves that persist across timeframes. Enters long when 4h price
# breaks above 1d Donchian high (20-period) with volume >1.5x average and price > 20-period SMA.
# Enters short when price breaks below 1d Donchian low with same conditions.
# Exits on mean reversion to the opposite Donchian level.
# Works in bull markets (buying breakouts) and bear markets (selling breakdowns).
# Designed for low trade frequency (~20-40 trades/year) to minimize drag.