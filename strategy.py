# 4h_1d_Donchian_Breakout_With_Trend_And_Volume
# Trend following with 1d EMA filter and 4h Donchian breakout
# Works in bull/bear: 1d EMA200 defines regime, 4h Donchian breakout triggers entries
# Volume filter ensures conviction
# Target: 20-40 trades/year per symbol

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian_Breakout_With_Trend_And_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d: EMA200 trend filter ===
    close_1d = df_1d['close'].values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # === 4h: Donchian channel and volume ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) - high/low of last 20 periods
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 50-period average)
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = volume / np.where(vol_ma50 > 0, vol_ma50, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema_val = ema200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(donch_high_val) or np.isnan(donch_low_val) or np.isnan(ema_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend with volume
            if (high_val > donch_high_val and
                close_val > ema_val and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in downtrend with volume
            elif (low_val < donch_low_val and
                  close_val < ema_val and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend change
            if (low_val < donch_low_val or
                close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend change
            if (high_val > donch_high_val or
                close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals