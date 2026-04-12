# 4h_1d_donchian_breakout_volume_atr
# Hypothesis: 4-hour Donchian breakout with volume confirmation and ATR volatility filter
# Works in bull/bear by using volatility-adjusted breakouts and volume confirmation to avoid false signals.
# Target: 25-50 trades/year (100-200 total over 4 years) to minimize fee drag.

name = "4h_1d_donchian_breakout_volume_atr"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Donchian and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channel (20-day high/low)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility filter (14-day ATR)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(high_1d, 1)))  # Using high for consistency
    tr3 = np.abs(np.subtract(low_1d, np.roll(low_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian levels and ATR to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above Donchian high with volume and volatility filter
        if (close[i] > donchian_high_aligned[i] and vol_confirm[i] and 
            atr_aligned[i] > 0 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below Donchian low with volume and volatility filter
        elif (close[i] < donchian_low_aligned[i] and vol_confirm[i] and 
              atr_aligned[i] > 0 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite Donchian level
        elif position == 1 and close[i] < donchian_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donchian_high_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals