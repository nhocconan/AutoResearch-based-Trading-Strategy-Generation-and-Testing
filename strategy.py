#!/usr/bin/env python3
"""
1d Williams Alligator with 1w EMA50 Trend Filter and Volume Spike
Hypothesis: Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) identifies trend absence/presence.
When aligned with 1w EMA50 trend and confirmed by volume spikes, this captures strong trending moves
while avoiding chop. Designed for 1d timeframe to target 7-25 trades/year (30-100 over 4 years)
by requiring confluence of Alligator alignment, 1w EMA50 trend, and volume confirmation.
Works in both bull (long when Lips>Teeth>Jaw above EMA50) and bear (short when Lips<Teeth<Jaw below EMA50).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams Alligator on primary timeframe (1d)
    # Jaw: 13-period SMMA (smoothed) of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    # SMMA: smoothed moving average (similar to Wilder's smoothing)
    median_price = (high + low) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    lips = np.roll(lips, 3)  # shift 3 bars forward
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max shift 8 + lips period 5 = 13) and EMA50
    start_idx = max(13, 50)  # Alligator setup, EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        # Lips < Teeth < Jaw = bearish alignment
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + trend + volume
            # Long: bullish Alligator alignment AND bullish bias AND volume spike
            long_entry = bullish_alignment and bullish_bias and vol_spike
            # Short: bearish Alligator alignment AND bearish bias AND volume spike
            short_entry = bearish_alignment and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: loss of bullish Alligator alignment OR loss of bullish bias
            if not bullish_alignment or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: loss of bearish Alligator alignment OR loss of bearish bias
            if not bearish_alignment or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0