#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with 12h volume confirmation and ATR trailing stop.
# Long when price breaks above Camarilla R1 AND 12h volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S1 AND 12h volume > 1.5x 20-period average.
# Exit on ATR-based trailing stop (2.5*ATR from extreme price) or opposite Camarilla break.
# Uses discrete position size 0.25. Works in both bull and bear markets by requiring
# volume confirmation and using symmetric breakout levels from daily pivot.
# Target: 75-150 total trades over 4 years (19-38/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Pivot Points (using prior 12h bar's OHLC) ===
    df_12h = get_htf_data(prices, '12h')
    # Prior completed 12h bar's OHLC
    pivot_high_12h = df_12h['high'].values
    pivot_low_12h = df_12h['low'].values
    pivot_close_12h = df_12h['close'].values
    pivot = (pivot_high_12h + pivot_low_12h + pivot_close_12h) / 3.0
    r1 = pivot + 1.1 * (pivot_high_12h - pivot_low_12h) / 12.0
    s1 = pivot - 1.1 * (pivot_high_12h - pivot_low_12h) / 12.0
    # Align to 6h timeframe (wait for 12h bar close)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # === 12h Volume Spike (volume > 1.5x 20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # === 6h ATR for trailing stop ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 50
    
    # Track position state, entry price, and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0  # highest high for long, lowest low for short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_6h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update extreme price (highest high)
            if price > extreme_price:
                extreme_price = price
            # ATR-based trailing stop: 2.5*ATR below extreme price
            if price < extreme_price - 2.5 * atr_val:
                exit_signal = True
            # Opposite breakout: price breaks below S1
            elif price < s1_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update extreme price (lowest low)
            if price < extreme_price:
                extreme_price = price
            # ATR-based trailing stop: 2.5*ATR above extreme price
            if price > extreme_price + 2.5 * atr_val:
                exit_signal = True
            # Opposite breakout: price breaks above R1
            elif price > r1_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 AND volume spike
            if price > r1_val and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_price = price
            
            # SHORT: Price breaks below S1 AND volume spike
            elif price < s1_val and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_VolumeSpike_ATRTrail_V1"
timeframe = "6h"
leverage = 1.0