#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray combo with 1w trend filter and volume confirmation
# Uses 1d primary timeframe for signal generation with Williams Alligator (JAW/TEETH/LIPS)
# Elder Ray (Bull Power/Bear Power) confirms momentum direction with EMA13
# 1w EMA50 trend filter provides higher timeframe bias (price > EMA50 for longs, < for shorts)
# Volume confirmation (1.5x 20-period average) filters for strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Williams Alligator identifies trend absence (all lines intertwined) vs trend presence (diverged lines)
# Elder Ray adds momentum confirmation to reduce false signals
# Works in both bull and bear markets by only trading in direction of 1w trend
# Combines trend-following (Alligator) with momentum (Elder Ray) for robust signals

name = "1d_WilliamsAlligator_ElderRay_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator (13,8,5) - SMMA with future shift
    # JAW (Blue): 13-period SMMA shifted 8 bars forward
    # TEETH (Red): 8-period SMMA shifted 5 bars forward  
    # LIPS (Green): 5-period SMMA shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First shifted values are invalid
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Elder Ray - Bull Power and Bear Power using EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator conditions: Lips > Teeth > Jaw (uptrend) OR Lips < Teeth < Jaw (downtrend)
            alligator_long = lips[i] > teeth[i] > jaw[i]
            alligator_short = lips[i] < teeth[i] < jaw[i]
            
            # Elder Ray conditions: Bull Power > 0 and rising OR Bear Power < 0 and falling
            # Simplified: Bull Power > 0 for long, Bear Power < 0 for short
            elder_long = bull_power[i] > 0
            elder_short = bear_power[i] < 0
            
            # Long: Alligator uptrend + Elder Ray bullish + volume spike + price > 1w EMA50
            if alligator_long and elder_long and volume_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + Elder Ray bearish + volume spike + price < 1w EMA50
            elif alligator_short and elder_short and volume_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses (Lips < Jaw) OR Elder Ray turns bearish (Bull Power < 0) OR price < 1w EMA50
            if lips[i] < jaw[i] or bull_power[i] < 0 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses (Lips > Jaw) OR Elder Ray turns bullish (Bear Power > 0) OR price > 1w EMA50
            if lips[i] > jaw[i] or bear_power[i] > 0 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals