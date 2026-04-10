#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and ATR-based trailing stop
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs smoothed
# - Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND 1d volume > 1.5x 20-period average
# - Short when Lips < Teeth < Jaw (bearish alignment) AND price < Lips AND 1d volume > 1.5x 20-period average
# - Exit via ATR trailing stop: 3 * ATR(14) from extreme price
# - Target: 12-37 trades/year on 12h (50-150 total over 4 years) to avoid fee drag
# - Williams Alligator catches trends early with smoothed SMAs, reducing whipsaw
# - 1d volume filter ensures participation from higher timeframe
# - ATR trailing stop manages risk without look-ahead

name = "12h_1d_williams_alligator_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume and its 20-period average for volume spike
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan, dtype=float)
    for i in range(19, len(vol_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    vol_spike_1d = np.full_like(vol_1d, False, dtype=bool)
    for i in range(20, len(vol_1d)):
        if not np.isnan(vol_ma_20[i]):
            vol_spike_1d[i] = vol_1d[i] > 1.5 * vol_ma_20[i]
    
    # Align 1d volume spike to 12h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Pre-compute ATR(14) for 12h trailing stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute Williams Alligator on 12h close
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods
    # Lips: 5-period SMMA smoothed by 3 periods
    def smma(arr, period):
        """Smoothed Moving Average (Wilder's smoothing)"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            result[period-1] = np.nanmean(arr[1:period+1])  # Simple average of first period values
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    jaw = smma(jaw_raw, 8)
    teeth = smma(teeth_raw, 5)
    lips = smma(lips_raw, 3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):  # Start after warmup for Alligator
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_now = close[i]
        high_now = high[i]
        low_now = low[i]
        jaw_now = jaw[i]
        teeth_now = teeth[i]
        lips_now = lips[i]
        atr_now = atr[i]
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # Convert back to boolean
        
        # Williams Alligator alignment conditions
        bullish_alignment = lips_now > teeth_now and teeth_now > jaw_now
        bearish_alignment = lips_now < teeth_now and teeth_now < jaw_now
        price_above_lips = close_now > lips_now
        price_below_lips = close_now < lips_now
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bullish alignment AND price above lips AND 1d volume spike
            if (bullish_alignment and price_above_lips and vol_spike):
                position = 1
                entry_price = close_now
                highest_since_entry = close_now
                lowest_since_entry = close_now
                signals[i] = 0.25
            # Short conditions: bearish alignment AND price below lips AND 1d volume spike
            elif (bearish_alignment and price_below_lips and vol_spike):
                position = -1
                entry_price = close_now
                highest_since_entry = close_now
                lowest_since_entry = close_now
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - update extremes and check trailing stop
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, high_now)
                lowest_since_entry = low_now  # Reset low for long (we trail from high)
                
                # ATR trailing stop: exit if price drops 3*ATR from highest since entry
                if close_now < highest_since_entry - 3.0 * atr_now:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, low_now)
                highest_since_entry = high_now  # Reset high for short (we trail from low)
                
                # ATR trailing stop: exit if price rises 3*ATR from lowest since entry
                if close_now > lowest_since_entry + 3.0 * atr_now:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals