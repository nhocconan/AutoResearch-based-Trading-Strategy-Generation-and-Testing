#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d volume confirmation and ATR stoploss.
# Long when price breaks above 4h Donchian(20) high AND 1d volume > 1.3x 20-period average.
# Short when price breaks below 4h Donchian(20) low AND 1d volume > 1.3x 20-period average.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Donchian break.
# Uses discrete position size 0.20. Session filter 08-20 UTC to reduce noise.
# Target: 60-150 total trades over 4 years (15-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_roll_aligned = align_htf_to_ltf(prices, df_4h, high_roll)
    low_roll_aligned = align_htf_to_ltf(prices, df_4h, low_roll)
    
    # === 1d Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.3 * vol_ma_1d_aligned)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high_4h).diff()
    tr2 = pd.Series(low_4h).diff().abs()
    tr3 = pd.Series(close[:len(high_4h)]).shift(1).diff().abs()  # Align close to 4h
    # Need to align TR calculation to 4h timeframe
    tr_4h = pd.concat([
        pd.Series(high_4h).diff(),
        pd.Series(low_4h).diff().abs(),
        pd.Series(close[:len(high_4h)]).shift(1).diff().abs()
    ], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, 14 for ATR)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(high_roll_aligned[i]) or np.isnan(low_roll_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below 4h Donchian low (opposite breakout)
            if price < low_roll_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above 4h Donchian high (opposite breakout)
            if price > high_roll_aligned[i]:
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
            # LONG: Price breaks above 4h Donchian high AND volume spike
            if price > high_roll_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below 4h Donchian low AND volume spike
            elif price < low_roll_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Donchian20_1dVolumeSpike_ATRStop_V1"
timeframe = "1h"
leverage = 1.0