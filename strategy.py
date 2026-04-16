#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla Pivot R1/S1 levels with volume spike confirmation.
# Long when price breaks above R1 with volume > 1.5x 20-period average.
# Short when price breaks below S1 with volume > 1.5x 20-period average.
# Exit when price reverts to the 1d VWAP (mean reversion to fair value).
# Uses discrete position size 0.25. Camarilla pivots provide intraday support/resistance
# levels derived from prior day's range, effective in ranging markets.
# Volume spike confirms breakout authenticity, reducing false signals.
# 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) and VWAP ===
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # VWAP = cumulative(price * volume) / cumulative(volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pv_1d = typical_price_1d * volume_1d
    cum_pv_1d = np.cumsum(pv_1d)
    cum_vol_1d = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_pv_1d, cum_vol_1d, out=np.full_like(cum_pv_1d, np.nan), where=cum_vol_1d!=0)
    
    # Calculate Camarilla levels from prior day's OHLC (shifted by 1 to avoid look-ahead)
    # We use the previous completed day's data for today's levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan  # First day has no prior
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1_1d = prev_close_1d + 1.1 * camarilla_range / 12.0
    s1_1d = prev_close_1d - 1.1 * camarilla_range / 12.0
    
    # === 1d Indicators: Volume Average for Spike Detection ===
    # 20-period average volume on 1d timeframe
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to primary timeframe (12h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # Sufficient for VWAP calculation and volume MA
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vwap = vwap_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = vol > 1.5 * vol_ma
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to VWAP (mean reversion)
            if abs(price - vwap) < 0.001 * vwap:  # Within 0.1% of VWAP
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to VWAP (mean reversion)
            if abs(price - vwap) < 0.001 * vwap:  # Within 0.1% of VWAP
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike
            if (price > r1) and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 with volume spike
            elif (price < s1) and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dCamarilla_R1S1_VolumeSpike_VWAPExit_V1"
timeframe = "12h"
leverage = 1.0