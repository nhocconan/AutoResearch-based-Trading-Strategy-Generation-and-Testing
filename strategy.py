#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Camarilla pivot levels with volume confirmation.
# Long when price breaks above weekly R4 AND 1d volume > 1.5x 20-period average.
# Short when price breaks below weekly S4 AND 1d volume > 1.5x 20-period average.
# Exit on opposite weekly Camarilla level (R3 for longs, S3 for shorts) or ATR stoploss (2*ATR).
# Uses discrete position size 0.25. Weekly pivots provide structural levels that work in both
# bull and bear markets; volume confirmation filters false breakouts. Target: 50-150 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w Indicators: Camarilla Pivot Levels ===
    df_1w = get_htf_data(prices, '1w')
    # Typical price for pivot calculation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    typical_price_vals = typical_price.values
    
    # Calculate weekly pivot and Camarilla levels
    pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    range_w = df_1w['high'].values - df_1w['low'].values
    
    # Camarilla levels
    R4 = pivot + (range_w * 1.1 / 2)
    R3 = pivot + (range_w * 1.1 / 4)
    S3 = pivot - (range_w * 1.1 / 4)
    S4 = pivot - (range_w * 1.1 / 2)
    
    # Align weekly levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # === 1d Indicators: Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for volume MA)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(S4_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr_6h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below weekly R3 (take profit)
            if price < R3_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above weekly S3 (take profit)
            if price > S3_aligned[i]:
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
            # LONG: Price breaks above weekly R4 AND volume spike
            if price > R4_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below weekly S4 AND volume spike
            elif price < S4_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WeeklyCamarilla_R4_S4_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0