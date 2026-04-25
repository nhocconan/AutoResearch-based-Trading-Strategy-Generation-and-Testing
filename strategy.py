#!/usr/bin/env python3
"""
12h Williams Alligator with 1d/1w Trend Filter and Volume Confirmation
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend direction and exhaustion.
In strong trends, lips > teeth > jaw (bull) or lips < teeth < jaw (bear).
We trade in the direction of the 1d/1w EMA50 trend only, confirmed by volume spikes.
Exit when Alligator lines re-cross (trend weakness) or volume dries up.
Designed for 12h timeframe to target 12-37 trades/year (50-150 over 4 years)
by requiring confluence of Alligator alignment, HTF trend, and volume confirmation.
Works in bull (long Alligator alignment) and bear (short Alligator alignment) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Primary 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Need sufficient HTF data for EMA50
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d and 1w EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams Alligator on 12h: SMAs of median price
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    # SMMA = smoothed moving average (like EMA but with alpha=1/period)
    median_price = (high + low) / 2.0
    
    # Calculate SMMA using EMA with alpha=1/period (approximation)
    def smma(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan, dtype=float)
        # First value: simple average
        result = np.full_like(series, np.nan, dtype=float)
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA[i] = (SMMA[i-1] * (period-1) + close[i]) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    smma_jaw = smma(median_price, 13)
    smma_teeth = smma(median_price, 8)
    smma_lips = smma(median_price, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(smma_jaw, 8)  # shifted 8 bars forward
    teeth = np.roll(smma_teeth, 5)  # shifted 5 bars forward
    lips = np.roll(smma_lips, 3)  # shifted 3 bars forward
    
    # Invalidate shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: current volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max shift 8) and EMA50
    start_idx = max(13, 50)  # Alligator jaw period, EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price above BOTH 1d and 1w EMA50 = bullish bias
        bullish_bias = (curr_close > ema_1d_aligned[i]) and (curr_close > ema_1w_aligned[i])
        # Trend filter: price below BOTH 1d and 1w EMA50 = bearish bias
        bearish_bias = (curr_close < ema_1d_aligned[i]) and (curr_close < ema_1w_aligned[i])
        
        # Alligator alignment: lips > teeth > jaw = bullish alignment
        bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Alligator alignment: lips < teeth < jaw = bearish alignment
        bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + HTF trend + volume
            # Long: bullish Alligator AND bullish HTF trend AND volume spike
            long_entry = bullish_alligator and bullish_bias and vol_spike
            # Short: bearish Alligator AND bearish HTF trend AND volume spike
            short_entry = bearish_alligator and bearish_bias and vol_spike
            
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
            # Exit: Alligator loses bullish alignment OR HTF trend turns bearish OR volume dries up
            if (not bullish_alligator) or (not bullish_bias) or (not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator loses bearish alignment OR HTF trend turns bullish OR volume dries up
            if (not bearish_alligator) or (not bearish_bias) or (not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1d1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0