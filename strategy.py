#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Long when price breaks above Donchian(20) high AND volume > 2.0x 20-period average.
# Short when price breaks below Donchian(20) low AND volume > 2.0x 20-period average.
# Exit when price retraces to Donchian midpoint OR ATR stoploss triggers.
# Uses discrete position size 0.25. Volume spike ensures institutional participation,
# Donchian provides objective breakout levels, ATR stoploss manages risk.
# Designed for fewer, higher-quality trades (~100/year) to minimize fee drag.
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # === 4h Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 30
    
    # Track position state and entry price for ATR stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        midpoint = donchian_mid[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls to Donchian midpoint OR 2*ATR stoploss
            if price <= midpoint or price <= entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises to Donchian midpoint OR 2*ATR stoploss
            if price >= midpoint or price >= entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND volume spike
            if price > upper_channel and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian(20) low AND volume spike
            elif price < lower_channel and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0