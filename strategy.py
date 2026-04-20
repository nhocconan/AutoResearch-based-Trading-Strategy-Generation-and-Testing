#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ATR stop
# - Long when price breaks above 20-period 12h high with volume > 1.5x 20-period 1d average
# - Short when price breaks below 20-period 12h low with volume > 1.5x 20-period 1d average
# - Exit when price returns to 10-period 12h moving average or ATR stop (2*ATR)
# - Uses 1d volume for confirmation (less noisy) and 12h for price action
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR for stop loss (using 1d data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Highest high over last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low over last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h 10-period moving average for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or \
           np.isnan(ma_10[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike
            if price > donchian_high[i] and vol > 1.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low + volume spike
            elif price < donchian_low[i] and vol > 1.5 * vol_ma_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to 10-period MA OR ATR stop hit (2*ATR)
            if price < ma_10[i] or price < entry_price - 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 10-period MA OR ATR stop hit (2*ATR)
            if price > ma_10[i] or price > entry_price + 2.0 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0