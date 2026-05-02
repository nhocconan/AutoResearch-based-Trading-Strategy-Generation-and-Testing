#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike strategy
# Uses Williams Alligator (13/8/5 SMAs) to identify trend direction and market structure
# Elder Ray (13-period EMA) measures bull/bear power relative to trend
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Primary timeframe: 1d for lower trade frequency and better signal quality
# HTF: 1w for major trend context
# Designed for 30-80 trades over 4 years (7-20/year) to minimize fee drag
# Works in bull markets via trend-following signals, in bear via reversal signals from Elder Ray extremes
# Alligator jaw/teeth/lips provide dynamic support/resistance for trend confirmation

name = "1d_WilliamsAlligator_ElderRay_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for major trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for major trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator components (13/8/5 periods)
    # Jaw (13-period SMA, shifted 8 bars)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMA, shifted 5 bars)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMA, shifted 3 bars)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate Elder Ray components (13-period EMA)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = lips[i] > teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment + Bull Power > 0 (strength) + volume confirm + price above 1w EMA50
            if bullish_alignment and bull_power[i] > 0 and volume_confirm[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + Bear Power < 0 (strength) + volume confirm + price below 1w EMA50
            elif bearish_alignment and bear_power[i] < 0 and volume_confirm[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish Alligator alignment OR Bear Power < 0 (trend weakness)
            if bearish_alignment or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment OR Bull Power > 0 (trend strength)
            if bullish_alignment or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals