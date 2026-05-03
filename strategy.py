#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Alligator's Jaw/Teeth/Lips for trend direction and entry signals.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation filters weak breakouts.
# Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe.
# Williams Alligator is effective in both trending and ranging markets via its convergence/divergence.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5) from prior completed 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 completed bar for prior
        return np.zeros(n)
    
    # Calculate prior completed 1d bar's median price for Alligator
    prior_median_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    prior_median_1d = np.roll(prior_median_1d, 1)  # Shift for prior bar
    prior_median_1d[0] = np.nan
    
    # Williams Alligator lines
    median_series = pd.Series(prior_median_1d)
    jaw = median_series.rolling(window=13, min_periods=13).mean().shift(8).values  # Jaw: 13-period, shifted 8
    teeth = median_series.rolling(window=8, min_periods=8).mean().shift(5).values   # Teeth: 8-period, shifted 5
    lips = median_series.rolling(window=5, min_periods=5).mean().shift(3).values    # Lips: 5-period, shifted 3
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(30) for stoploss (using 6h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=30, min_periods=30, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 30-bar average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Alligator conditions
        # Lips above Teeth above Jaw = bullish alignment
        bullish_align = lips_val > teeth_val and teeth_val > jaw_val
        # Lips below Teeth below Jaw = bearish alignment
        bearish_align = lips_val < teeth_val and teeth_val < jaw_val
        
        # Entry conditions
        # Long: bullish alignment + price above Lips + above 1d EMA34 + volume spike
        long_entry = bullish_align and (close[i] > lips_val) and (close[i] > ema_trend) and vol_spike
        # Short: bearish alignment + price below Lips + below 1d EMA34 + volume spike
        short_entry = bearish_align and (close[i] < lips_val) and (close[i] < ema_trend) and vol_spike
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals