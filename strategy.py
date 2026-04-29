#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike and ATR-based position sizing
# Donchian breakout captures strong momentum moves, volume confirms institutional interest
# ATR stoploss limits downside during volatile periods (e.g., 2022 crash)
# Works in bull/bear: volume filter avoids fakeouts, ATR adapts to volatility regimes
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_VolumeSpike_ATR_Stop_v1"
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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    # ATR(14) for volatility-based stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = max(30, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_30[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            if curr_volume_confirm:
                # Bullish breakout: price closes above upper Donchian band
                if curr_close > high_20[i]:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    atr_stop = entry_price - 2.5 * curr_atr
                # Bearish breakout: price closes below lower Donchian band
                elif curr_close < low_20[i]:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    atr_stop = entry_price + 2.5 * curr_atr
        
        elif position == 1:  # Long position
            # Trail stop: exit if price drops below ATR-based stop
            if curr_low < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                # Update stop if price moves favorably
                new_stop = curr_close - 2.5 * curr_atr
                if new_stop > atr_stop:
                    atr_stop = new_stop
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Trail stop: exit if price rises above ATR-based stop
            if curr_high > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                # Update stop if price moves favorably
                new_stop = curr_close + 2.5 * curr_atr
                if new_stop < atr_stop:
                    atr_stop = new_stop
                signals[i] = -0.30
    
    return signals