#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above 20-period Donchian high with volume > 1.5x 20-period MA.
# Short when price breaks below 20-period Donchian low with volume > 1.5x 20-period MA.
# Uses ATR(14) for dynamic stoploss: exit long if price drops 2.0*ATR below entry, exit short if price rises 2.0*ATR above entry.
# Position sizing: 0.30 (30% of capital) to balance risk and return.
# Volume confirmation filters breakouts for institutional participation.
# ATR stoploss manages risk in both bull and bear markets.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
# Works in both bull and bear markets: breakouts capture trends, volume confirms validity, ATR stoploss limits drawdowns.

name = "4h_Donchian20_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high and low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for ATR-based stoploss
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Breakout conditions
        breakout_high = close_val > donchian_high[i]
        breakout_low = close_val < donchian_low[i]
        
        if position == 0:
            # Look for breakout entries with volume confirmation
            if breakout_high and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif breakout_low and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: trail stoploss or hold
            signals[i] = 0.30
            # ATR-based stoploss: exit if price drops 2.0*ATR below entry
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short position: trail stoploss or hold
            signals[i] = -0.30
            # ATR-based stoploss: exit if price rises 2.0*ATR above entry
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals