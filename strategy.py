#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw, Teeth, Lips) on 6h for trend identification and entry signals
# 1d EMA50 acts as higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.5x average) ensures strong participation and reduces false signals
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Williams Alligator is effective in both trending and ranging markets when combined with HTF filter
# Prioritizes BTC/ETH performance with SOL as secondary

name = "6h_WilliamsAlligator_1dEMA50_Trend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: SMAs of median price (typical price) with different periods
    typical_price = (high + low + close) / 3.0
    
    # Jaw: 13-period SMMA shifted 8 bars
    jaw_raw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth_raw = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips: 5-period SMMA shifted 3 bars
    lips_raw = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid due to shift
    
    # Align Alligator lines to 6h timeframe (already on 6h, but ensure proper alignment)
    # Since we calculated on 6h data directly, no HTF alignment needed for Alligator
    # But we'll keep the variables for consistency
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams Alligator signals with 1d trend filter
        # Lips above Teeth above Jaw = bullish alignment
        # Lips below Teeth below Jaw = bearish alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Enter long: bullish alignment + volume spike + price above 1d EMA50 (uptrend)
            if bullish_alignment and volume_spike and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + volume spike + price below 1d EMA50 (downtrend)
            elif bearish_alignment and volume_spike and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment (reversal) OR price below 1d EMA50 (trend change)
            if bearish_alignment or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment (reversal) OR price above 1d EMA50 (trend change)
            if bullish_alignment or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals