#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1d trend filter and volume confirmation
# Long when price breaks above Alligator Jaw AND 1d close > 1d EMA34 (uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below Alligator Jaw AND 1d close < 1d EMA34 (downtrend) AND volume > 1.5x 20 EMA
# Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# Breakout defined as price crossing above/below Jaw with Teeth and Lips aligned in same direction
# Uses 12h for primary timeframe (lower trade frequency), 1d for trend filter to avoid counter-trend trades
# Discrete sizing (0.25) to balance profit potential and risk management
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends

name = "12h_WilliamsAlligator_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d data for Williams Alligator and trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (SMMA = Smoothed Moving Average)
    # SMMA formula: SMMA_t = (SMMA_{t-1} * (period-1) + close_t) / period
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price_1d = (high_1d + low_1d) / 2
    jaw_raw = smma(median_price_1d, 13)
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Alligator Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = smma(median_price_1d, 8)
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid after shift
    
    # Alligator Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = smma(median_price_1d, 5)
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid after shift
    
    # Alligator alignment: 
    # Uptrend: Lips > Teeth > Jaw (all aligned upward)
    # Downtrend: Lips < Teeth < Jaw (all aligned downward)
    alligator_uptrend = (lips > teeth) & (teeth > jaw)
    alligator_downtrend = (lips < teeth) & (teeth < jaw)
    
    # Breakout conditions: price breaks above/below Jaw with alignment
    bullish_breakout = (close_1d > jaw) & alligator_uptrend
    bearish_breakout = (close_1d < jaw) & alligator_downtrend
    
    # Align 1d indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    bullish_breakout_aligned = align_htf_to_ltf(prices, df_1d, bullish_breakout.astype(float))
    bearish_breakout_aligned = align_htf_to_ltf(prices, df_1d, bearish_breakout.astype(float))
    alligator_uptrend_aligned = align_htf_to_ltf(prices, df_1d, alligator_uptrend.astype(float))
    alligator_downtrend_aligned = align_htf_to_ltf(prices, df_1d, alligator_downtrend.astype(float))
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 12h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume spike filter (20-period volume EMA on 12h data)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(bullish_breakout_aligned[i]) or 
            np.isnan(bearish_breakout_aligned[i]) or np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish Alligator breakout AND 1d uptrend AND volume spike
            if (bullish_breakout_aligned[i] > 0.5 and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Alligator breakout AND 1d downtrend AND volume spike
            elif (bearish_breakout_aligned[i] > 0.5 and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish Alligator breakout OR 1d trend changes to downtrend
            if (bearish_breakout_aligned[i] > 0.5 or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish Alligator breakout OR 1d trend changes to uptrend
            if (bullish_breakout_aligned[i] > 0.5 or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals