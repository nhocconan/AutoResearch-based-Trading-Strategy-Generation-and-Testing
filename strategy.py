#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.3x 20-period average volume.
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.3x 20-period average volume.
# Exit when price crosses back below/above Donchian(20) midline.
# Uses Donchian for breakout structure, daily EMA for trend filter, volume for confirmation.
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
    
    # Donchian channels (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend filter
    daily_ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Daily 20-period average volume for confirmation
    vol_ma_20d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure Donchian is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_ema50_aligned[i]) or np.isnan(vol_ma_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        midline = donchian_mid[i]
        daily_ema = daily_ema50_aligned[i]
        vol_ma = vol_ma_20d_aligned[i]
        vol = volume[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long entry: break above upper band + above daily EMA + volume confirmation
            if price > upper and price > daily_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band + below daily EMA + volume confirmation
            elif price < lower and price < daily_ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midline
            if price < midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midline
            if price > midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals