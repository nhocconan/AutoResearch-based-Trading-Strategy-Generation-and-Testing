#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h volume confirmation and ATR stoploss.
# Long when price breaks above Camarilla R1 AND 4h volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S1 AND 4h volume > 1.5x 20-period average.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Camarilla break.
# Uses discrete position size 0.20. Session filter (08-20 UTC) to reduce noise.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Works in both bull and bear markets by requiring volume confirmation and using
# symmetric Camarilla levels derived from prior 4h session.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Points (R1, S1) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate prior 4h bar's Camarilla levels
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    hl_range_4h = high_4h - low_4h
    r1_4h = close_4h + (1.1 * hl_range_4h / 12)
    s1_4h = close_4h - (1.1 * hl_range_4h / 12)
    
    # Align to 1h timeframe (completed 4h bar only)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike_4h = vol_4h > (1.5 * vol_ma_4h_aligned)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # === 4h ATR for stoploss ===
    tr1_4h = pd.Series(high_4h).diff()
    tr2_4h = pd.Series(low_4h).diff().abs()
    tr3_4h = pd.Series(close_4h).shift(1).diff().abs()
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for 4h)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(volume_spike_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike_4h_aligned[i]
        atr_val = atr_4h_aligned[i]
        r1 = r1_4h_aligned[i]
        s1 = s1_4h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1 (opposite breakout)
            if price < s1:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1 (opposite breakout)
            if price > r1:
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
            # LONG: Price breaks above Camarilla R1 AND volume spike
            if price > r1 and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike
            elif price < s1 and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Camarilla_R1S1_4hVolumeSpike_ATRStop_V1"
timeframe = "1h"
leverage = 1.0