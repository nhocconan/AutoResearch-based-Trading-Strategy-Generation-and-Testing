#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout + 1d volume spike + ATR filter
    # Long when price breaks above Donchian(20) high AND volume > 1.5x 20-period avg AND ATR(14) < ATR(50) (low vol regime)
    # Short when price breaks below Donchian(20) low AND volume > 1.5x 20-period avg AND ATR(14) < ATR(50)
    # Exit when price touches opposite Donchian band or ATR spikes (vol expansion)
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via ATR regime filter avoiding high volatility whipsaws.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume ratio (current / 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0.0)
    
    # Calculate ATR(14) and ATR(50) for volatility regime filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr50 > 0, atr14 / atr50, 1.0)  # Low vol when ratio < 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # Start after warmup period for longest indicator
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation and low volatility filter
        breakout_up = close[i] > donchian_high[i-1] and vol_ratio[i] > 1.5 and atr_ratio[i] < 1.0
        breakout_down = close[i] < donchian_low[i-1] and vol_ratio[i] > 1.5 and atr_ratio[i] < 1.0
        
        # Exit conditions: touch opposite band or volatility expansion
        exit_long = close[i] < donchian_low[i] or atr_ratio[i] > 1.2
        exit_short = close[i] > donchian_high[i] or atr_ratio[i] > 1.2
        
        # Entry logic
        if breakout_up and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_down and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit logic
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_donchian_breakout_volume_atr_v1"
timeframe = "12h"
leverage = 1.0