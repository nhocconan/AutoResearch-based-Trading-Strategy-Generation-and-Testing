#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
# Long when price breaks above 20-period high AND volume > 1.5x 20-period average volume AND ATR(14) > 0.
# Short when price breaks below 20-period low AND volume > 1.5x 20-period average volume AND ATR(14) > 0.
# Exit when price retraces to 10-period Donchian midpoint OR ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture strong breakouts in trending markets with volume confirmation.
# Works in both bull and bear markets by requiring volume confirmation and avoiding false breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channels ===
    # 20-period high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # 10-period high/low for exit (midpoint)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (high_10 + low_10) / 2.0
    
    # === 4h Indicators: ATR(14) for stoploss and volume filter ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 20
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid_10[i]) or
            np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price retraces to 10-period Donchian midpoint
            if price <= donchian_mid_10[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price retraces to 10-period Donchian midpoint
            if price >= donchian_mid_10[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 20-period high AND volume spike
            if price > high_20[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below 20-period low AND volume spike
            elif price < low_20[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0