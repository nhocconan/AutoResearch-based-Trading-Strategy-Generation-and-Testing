#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR-based stoploss.
# Long when price breaks above 20-period high AND volume > 1.5x 20-period average.
# Short when price breaks below 20-period low AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Designed to capture strong momentum moves while avoiding chop.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag and maximize edge.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channel (20) ===
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_channel = high_ma.values
    lower_channel = low_ma.values
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for volume confirmation (more reliable than 12h volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for volume calculation
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # === ATR for stoploss (14-period) ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_values = atr.values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ATR, 20 for channels/volume)
    warmup = 40
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(atr_values[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        vol_spike_1d = bool(volume_spike_1d_aligned[i])
        atr_val = atr_values[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1 and entry_price > 0:  # Long position
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1 and entry_price > 0:  # Short position
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC (channel re-entry) ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price re-enters the channel (below upper band)
            if price < upper_channel[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price re-enters the channel (above lower band)
            if price > lower_channel[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper channel AND 1d volume spike
            if price > upper_channel[i] and vol_spike_1d:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below lower channel AND 1d volume spike
            elif price < lower_channel[i] and vol_spike_1d:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ATRStop_V1"
timeframe = "12h"
leverage = 1.0