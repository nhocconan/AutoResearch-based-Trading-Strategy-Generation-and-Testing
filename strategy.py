#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# Donchian channels capture breakout momentum; volume >1.5x confirms participation
# ATR stoploss limits drawdown in bear markets; discrete sizing (0.25) reduces fee churn
# Works in bull/bear: breakouts catch strong moves, volume filter avoids false signals, ATR stop manages risk
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_Donchian20_Breakout_VolumeConfirm_ATRStop_v1"
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
    
    # Calculate ATR for volatility and stoploss (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(20, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish entry: price breaks above Donchian upper + volume confirmation
            if curr_high > highest_high[i-1] and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = atr[i]
            # Bearish entry: price breaks below Donchian lower + volume confirmation
            elif curr_low < lowest_low[i-1] and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = atr[i]
        
        elif position == 1:  # Long position
            # ATR-based stoploss: exit if price drops below entry - 2.0 * ATR
            if curr_close < entry_price - 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Donchian lower
            elif curr_low < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # ATR-based stoploss: exit if price rises above entry + 2.0 * ATR
            if curr_close > entry_price + 2.0 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Donchian upper
            elif curr_high > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals