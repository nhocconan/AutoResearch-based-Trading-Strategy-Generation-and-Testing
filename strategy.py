#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator strategy with 1d trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend phases via SMAs
# When all three lines are entwined (no clear order) = sleeping (range)
# When Jaw > Teeth > Lips = uptrend (bullish alignment)
# When Jaw < Teeth < Lips = downtrend (bearish alignment)
# Enter on trend confirmation with volume spike (>2.0x 24-period average)
# Exit when Alligator starts to sleep (lines re-entwine) or trend filter fails
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaw
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 4h data
    close_4h = df_4h['close'].values
    # Jaw: 13-period SMMA (smoothed moving average) of median price, shifted 8 bars
    median_price_4h = (high_4h := (df_4h['high'].values + df_4h['low'].values) / 2)
    jaw_raw = pd.Series(median_price_4h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = pd.Series(median_price_4h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = pd.Series(median_price_4h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid due to shift
    
    # Align Alligator lines to 4h timeframe (already aligned since calculated on 4h)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*4h = 96h = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(13, 24)  # warmup for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Check Alligator alignment
        bullish_alignment = curr_jaw > curr_teeth and curr_teeth > curr_lips
        bearish_alignment = curr_jaw < curr_teeth and curr_teeth < curr_lips
        sleeping = not (bullish_alignment or bearish_alignment)  # lines entwined
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and clear Alligator alignment
            if curr_volume_spike and not sleeping:
                # Bullish entry: bullish alignment with price above 1d EMA34
                if bullish_alignment and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: bearish alignment with price below 1d EMA34
                elif bearish_alignment and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Alligator starts to sleep (lines re-entwine) OR price crosses below 1d EMA34
            if sleeping or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator starts to sleep (lines re-entwine) OR price crosses above 1d EMA34
            if sleeping or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals