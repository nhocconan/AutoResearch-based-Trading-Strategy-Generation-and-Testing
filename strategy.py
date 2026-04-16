#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and ATR trailing stop.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 20-period average.
# Exit on ATR-based trailing stop (2.5*ATR from extreme price) or opposite pivot break.
# Uses discrete position size 0.25. Volume confirmation reduces false breakouts.
# ATR trailing stop allows trends to run while limiting drawdown.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Calculate from previous completed bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar fallback
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    camarilla_r1 = pivot + (range_hl * 1.1 / 12.0)
    camarilla_s1 = pivot - (range_hl * 1.1 / 12.0)
    
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
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 50
    
    # Track position state, entry price, and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0  # highest price for long, lowest for short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update extreme price (highest reached)
            if price > extreme_price:
                extreme_price = price
            # Exit if price breaks below Camarilla S1 (opposite pivot)
            if price < camarilla_s1[i]:
                exit_signal = True
            # ATR-based trailing stop: 2.5*ATR below extreme price
            elif price < extreme_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update extreme price (lowest reached)
            if price < extreme_price:
                extreme_price = price
            # Exit if price breaks above Camarilla R1 (opposite pivot)
            if price > camarilla_r1[i]:
                exit_signal = True
            # ATR-based trailing stop: 2.5*ATR above extreme price
            elif price > extreme_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike
            if price > camarilla_r1[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike
            elif price < camarilla_s1[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike_ATRTrail_V1"
timeframe = "4h"
leverage = 1.0