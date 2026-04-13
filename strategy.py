#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR filter.
    # Long when price breaks above 20-period high with volume spike and ATR > ATR MA.
    # Short when price breaks below 20-period low with volume spike and ATR > ATR MA.
    # Exit when price crosses 10-period EMA in opposite direction.
    # Uses discrete size 0.25 to minimize fee churn.
    # Target: 75-200 total trades over 4 years (19-50/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate indicators (vectorized)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Donchian channels (20-period)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # EMA(10) for exit
    ema_10 = close_s.ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # ATR(14) for volatility filter
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation (20-period mean)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_10[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period mean
        volume_confirmation = volume[i] > 1.5 * volume_ma[i]
        
        # ATR filter: current ATR > 20-period ATR mean (ensures sufficient volatility)
        atr_filter = atr[i] > atr_ma[i]
        
        # Entry conditions: Donchian breakout with volume and ATR confirmation
        long_entry = (close[i] > donchian_high[i] and volume_confirmation and atr_filter)
        short_entry = (close[i] < donchian_low[i] and volume_confirmation and atr_filter)
        
        # Exit conditions: price crosses 10-period EMA in opposite direction
        long_exit = (position == 1 and close[i] < ema_10[i])
        short_exit = (position == -1 and close[i] > ema_10[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif long_exit:
            position = 0
            signals[i] = 0.0
        elif short_exit:
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

name = "4h_donchian_breakout_volume_atr_filter_v1"
timeframe = "4h"
leverage = 1.0