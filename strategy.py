#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 1.3x 12h average volume
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 1.3x 12h average volume
# Exit when price returns to Donchian midpoint or reverses with volume confirmation
# Uses Donchian for trend structure, EMA for higher timeframe trend filter, volume for breakout confirmation.
# Target: 20-50 trades/year per symbol.
name = "4h_Donchian_Breakout_Trend_Volume"
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
    
    # Get 12-hour data for Donchian channels and EMA
    df_12h = get_htf_data(prices, '12h')
    
    # Donchian channels (20-period high/low)
    high_20 = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donchian_high = high_20
    donchian_low = low_20
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12-period EMA for trend filter
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period average volume for confirmation
    vol_ma_20 = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure Donchian and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        d_high = donchian_high_aligned[i]
        d_low = donchian_low_aligned[i]
        d_mid = donchian_mid_aligned[i]
        ema = ema_50_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long entry: break above Donchian high + above EMA50 + volume confirmation
            if price > d_high and price > ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: break below Donchian low + below EMA50 + volume confirmation
            elif price < d_low and price < ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint OR breaks below low with volume
            if price < d_mid or (price < d_low and volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint OR breaks above high with volume
            if price > d_mid or (price > d_high and volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals