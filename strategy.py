#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Keltner Channel breakout with volume confirmation.
# Long when price breaks above upper KC(20,2) + volume > 1.8x 20-period median volume.
# Short when price breaks below lower KC(20,2) + volume > 1.8x 20-period median volume.
# Uses discrete position size 0.25. Exits when price returns to middle KC (EMA20) or when volume drops below median.
# Keltner Channel uses ATR for adaptive width, making it more responsive to volatility changes than Bollinger Bands.
# Volume confirmation ensures breakout has participation. 6h timeframe targets 12-37 trades/year to minimize fee drag.
# Works in both bull and bear markets by capturing volatility expansion breakouts with institutional volume.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Keltner Channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # === 1d Indicators: EMA(20) for middle line ===
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 1d Indicators: ATR(10) for channel width ===
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    true_range_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_10_1d = pd.Series(true_range_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Upper and Lower Keltner Channels (20,2)
    upper_kc = ema_20 + (2 * atr_10_1d)
    lower_kc = ema_20 - (2 * atr_10_1d)
    middle_kc = ema_20  # EMA20
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (6h)
    upper_kc_aligned = align_htf_to_ltf(prices, df_1d, upper_kc)
    lower_kc_aligned = align_htf_to_ltf(prices, df_1d, lower_kc)
    middle_kc_aligned = align_htf_to_ltf(prices, df_1d, middle_kc)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20)  # EMA20 needs 20, volume median needs 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_kc_aligned[i]) or np.isnan(lower_kc_aligned[i]) or np.isnan(middle_kc_aligned[i]) or
            np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        upper_kc_val = upper_kc_aligned[i]
        lower_kc_val = lower_kc_aligned[i]
        middle_kc_val = middle_kc_aligned[i]
        vol_median = vol_median_aligned[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.8x median volume
        volume_spike = current_vol_1d > (vol_median * 1.8)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to middle KC OR volume drops below median
            if (price <= middle_kc_val) or (current_vol_1d < vol_median):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to middle KC OR volume drops below median
            if (price >= middle_kc_val) or (current_vol_1d < vol_median):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper KC + volume spike
            if (price > upper_kc_val) and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower KC + volume spike
            elif (price < lower_kc_val) and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_1dKeltnerChannelBreakout_VolumeSpike1.8x_EXITmiddleKC_VolumeBelowMedian_v1"
timeframe = "6h"
leverage = 1.0