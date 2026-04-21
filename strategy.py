#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) shifted forward.
# In trending markets: lines are separated and ordered (Lips > Teeth > Jaw for uptrend).
# In ranging markets: lines intertwine and converge.
# Use 1d EMA34 as trend filter: only take Alligator signals when price > EMA34 (uptrend) or < EMA34 (downtrend).
# Volume confirmation: current volume > 1.5x 20-period average.
# Designed for 12h timeframe to capture medium-term trends with low trade frequency.
# Williams Alligator helps avoid whipsaws in ranging markets by requiring clear separation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (13, 8, 5 period SMAs with forward shift)
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    close = prices['close'].values
    
    # Smoothed Moving Average (SMMA) - same as RMA/Wilder's MA
    def smma(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA components
    jaw_raw = smma(close, 13)  # 13-period
    teeth_raw = smma(close, 8)  # 8-period
    lips_raw = smma(close, 5)   # 5-period
    
    # Apply forward shifts (Alligator specific)
    jaw = np.roll(jaw_raw, 8)   # shift 8 bars forward
    teeth = np.roll(teeth_raw, 5) # shift 5 bars forward
    lips = np.roll(lips_raw, 3)   # shift 3 bars forward
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator conditions
        # Uptrend: Lips > Teeth > Jaw (all separated and ordered)
        # Downtrend: Jaw > Teeth > Lips (all separated and ordered)
        # Convergence/ranging: lines intertwine
        lips_gt_teeth = lips[i] > teeth[i]
        teeth_gt_jaw = teeth[i] > jaw[i]
        jaw_gt_teeth = jaw[i] > teeth[i]
        teeth_gt_lips = teeth[i] > lips[i]
        
        is_uptrend_aligned = lips_gt_teeth and teeth_gt_jaw
        is_downtrend_aligned = jaw_gt_teeth and teeth_gt_lips
        
        # 1d EMA34 trend filter
        price = prices['close'].iloc[i]
        ema_filter = ema_34_1d_aligned[i]
        is_uptrend_1d = price > ema_filter
        is_downtrend_1d = price < ema_filter
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume = prices['volume'].iloc[i]
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for Alligator alignment with 1d trend and volume confirmation
            if is_uptrend_aligned and is_uptrend_1d and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif is_downtrend_aligned and is_downtrend_1d and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: when Alligator lines converge or trend changes
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when: Alligator loses uptrend alignment OR 1d trend turns down
                if not (is_uptrend_aligned and is_uptrend_1d):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when: Alligator loses downtrend alignment OR 1d trend turns up
                if not (is_downtrend_aligned and is_downtrend_1d):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34Trend_Volume"
timeframe = "12h"
leverage = 1.0