#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with 12h volume confirmation and ATR trailing stop.
# Long when price breaks above Camarilla R1 AND 12h volume > 1.2x 20-period average.
# Short when price breaks below Camarilla S1 AND 12h volume > 1.2x 20-period average.
# Exit on ATR-based trailing stop (2.5*ATR from extreme price) or opposite Camarilla break.
# Uses discrete position size 0.25. Works in both bull and bear markets by requiring
# volume confirmation and using symmetric breakout levels.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Camarilla levels calculated from previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # === 12h Indicators: Volume Spike (volume > 1.2x 20-period average) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.2 * vol_ma_12h_aligned)
    
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
    extreme_price = 0.0  # tracks highest price for long, lowest for short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_6h_raw[i]) or not session_filter[i]):
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
            # Update extreme price (highest close since entry)
            if price > extreme_price:
                extreme_price = price
            # Exit if price breaks below Camarilla S1 (opposite breakout)
            if price < s1[i]:
                exit_signal = True
            # ATR-based trailing stop: 2.5*ATR below extreme price
            elif price < extreme_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update extreme price (lowest close since entry)
            if price < extreme_price:
                extreme_price = price
            # Exit if price breaks above Camarilla R1 (opposite breakout)
            if price > r1[i]:
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
            if price > r1[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike
            elif price < s1[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_12hVolumeSpike_ATRTrail_V1"
timeframe = "6h"
leverage = 1.0