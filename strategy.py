#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ATR-based stop.
# Long when price breaks above Camarilla R4 level AND 1d volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S4 level AND 1d volume > 1.5x 20-period average.
# Uses ATR(14) for dynamic stoploss (signal → 0 when price moves against position by 2*ATR).
# Designed to capture strong breakouts in both bull and bear markets with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_values = atr.values
    
    # === 6h Indicators: Camarilla pivot levels (based on previous bar) ===
    # Camarilla levels calculated from previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else close[0]
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else high[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else low[0]
    
    # Calculate pivot point and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + (range_hl * 1.1 / 2.0)
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ATR, 20 for volume MA)
    warmup = 40
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_values[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Stoploss: price drops below entry_price - 2*ATR
            if price < entry_price - 2.0 * atr_values[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Stoploss: price rises above entry_price + 2*ATR
            if price > entry_price + 2.0 * atr_values[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above R4 AND 1d volume spike
            if price > r4[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below S4 AND 1d volume spike
            elif price < s4[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dVolumeSpike_ATRStop_V1"
timeframe = "6h"
leverage = 1.0