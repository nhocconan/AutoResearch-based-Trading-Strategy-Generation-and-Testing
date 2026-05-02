#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with weekly trend filter
# Uses Williams Alligator (jaw/teeth/lips) to identify trend absence (alligator sleeping)
# Elder Ray (Bull/Bear Power) to measure trend strength relative to EMA13
# Weekly EMA50 as regime filter: only trade in direction of weekly trend
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Designed for low trade frequency (target: 50-150 total trades over 4 years)
# Works in bull markets via trend-following entries, in bear via alligator sleep filter avoiding whipsaws

name = "6h_WilliamsAlligator_ElderRay_1wEMA50_Trend_VolumeConfirm_v1"
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
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator (6h timeframe)
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate Elder Ray Power (6h timeframe)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Alligator sleeping condition: all three lines intertwined (market ranging)
        # Alligator awake: lines are separated and ordered
        alligator_sleeping = (
            (jaw[i] > teeth[i] * 0.999 and jaw[i] < teeth[i] * 1.001) or  # jaw ~ teeth
            (teeth[i] > lips[i] * 0.999 and teeth[i] < lips[i] * 1.001) or  # teeth ~ lips
            (jaw[i] > lips[i] * 0.999 and jaw[i] < lips[i] * 1.001)  # jaw ~ lips
        )
        
        # Alligator awake and trending up: lips > teeth > jaw
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Alligator awake and trending down: jaw > teeth > lips
        alligator_down = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator awake AND trending up + Bull Power > 0 + price > weekly EMA50 + volume confirm
            if alligator_up and not alligator_sleeping and bull_power[i] > 0 and close[i] > ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator awake AND trending down + Bear Power < 0 + price < weekly EMA50 + volume confirm
            elif alligator_down and not alligator_sleeping and bear_power[i] < 0 and close[i] < ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator starts sleeping OR Bear Power turns negative OR price crosses below weekly EMA50
            if alligator_sleeping or bear_power[i] < 0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator starts sleeping OR Bull Power turns positive OR price crosses above weekly EMA50
            if alligator_sleeping or bull_power[i] > 0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals