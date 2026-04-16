#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ATR-based trailing stop.
# Uses discrete position sizing (0.30) to minimize fee churn. Volume confirmation reduces false breakouts.
# ATR trailing stop (3*ATR from extreme) lets winners run while controlling risk.
# Designed to work in both bull and bear markets by requiring volume spike and using symmetric rules.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 4h ATR for trailing stop ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state, entry price, and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0  # highest high for long, lowest low for short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # Current values
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update extreme price (highest high)
            if high_price > extreme_price:
                extreme_price = high_price
            # ATR-based trailing stop: 3*ATR below extreme
            if price < extreme_price - 3.0 * atr_val:
                exit_signal = True
            # Optional: exit on opposite Donchian break (uncomment if needed)
            # elif price < low_roll[i]:
            #     exit_signal = True
        
        elif position == -1:  # Short position
            # Update extreme price (lowest low)
            if low_price < extreme_price:
                extreme_price = low_price
            # ATR-based trailing stop: 3*ATR above extreme
            if price > extreme_price + 3.0 * atr_val:
                exit_signal = True
            # Optional: exit on opposite Donchian break (uncomment if needed)
            # elif price > high_roll[i]:
            #     exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike
            if price > high_roll[i] and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = price
                extreme_price = price
            
            # SHORT: Price breaks below Donchian low AND volume spike
            elif price < low_roll[i] and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = price
                extreme_price = price
        
        else:
            # Maintain position
            signals[i] = position * 0.30
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ATRTrail_V1"
timeframe = "4h"
leverage = 1.0