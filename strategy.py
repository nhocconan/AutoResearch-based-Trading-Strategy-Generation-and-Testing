#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d volume confirmation and ATR-based trailing stop.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 24-period average.
# Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 24-period average.
# Uses ATR(14) trailing stop: exit long when price drops 2.5*ATR from highest high since entry,
# exit short when price rises 2.5*ATR from lowest low since entry.
# Designed to capture strong intraday momentum with volume confirmation while avoiding whipsaws.
# Works in both bull and bear markets by requiring volume confirmation and using ATR stops.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_upper = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_lower = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 24-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=24, min_periods=24).mean().values
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
    
    # Warmup: ensure all indicators are valid (max 24 periods needed for volume MA)
    warmup = 50
    
    # Track position state and extreme prices for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_upper[i]) or np.isnan(camarilla_lower[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_raw[i]) or
            not session_filter[i]):
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
            # Update highest high since entry
            highest_high = max(highest_high, high[i])
            # Exit if price drops 2.5*ATR from highest high
            if price < highest_high - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low = min(lowest_low, low[i])
            # Exit if price rises 2.5*ATR from lowest low
            if price > lowest_low + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            highest_high = 0.0
            lowest_low = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike
            if price > camarilla_upper[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_high = high[i]
                lowest_low = low[i]
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike
            elif price < camarilla_lower[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                highest_high = high[i]
                lowest_low = low[i]
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0