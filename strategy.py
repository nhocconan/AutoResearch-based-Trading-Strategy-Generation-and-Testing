#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator with 1d Elder Ray and volume confirmation
    # Williams Alligator identifies trend direction via SMAs (Jaw/Teeth/Lips)
    # Elder Ray (Bull/Bear Power) confirms trend strength using EMA13
    # Volume filter ensures institutional participation
    # Session filter (08-20 UTC) avoids low-liquidity periods
    # Target: 15-30 trades/year to minimize fee drag on 12h timeframe
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator (based on previous 12h candle)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: Jaw (13-period SMMA, 8-shift), Teeth (8-period SMMA, 5-shift), Lips (5-period SMMA, 3-shift)
    # SMMA = smoothed moving average (EMA with alpha = 1/period)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        alpha = 1.0 / period
        res = np.full_like(arr, np.nan, dtype=float)
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = (arr[i] * alpha) + (res[i-1] * (1 - alpha))
        return res
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Shift as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data for Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter (24-period on 12h, approx 20-day equivalent)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma24  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma24[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + volume spike
            if (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and
                bull_power_aligned[i] > 0 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) + Bear Power > 0 + volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and
                  bear_power_aligned[i] > 0 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines reverse or Elder Power changes sign
            if position == 1:
                if (lips_aligned[i] < teeth_aligned[i] or bull_power_aligned[i] <= 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (jaw_aligned[i] < teeth_aligned[i] or bear_power_aligned[i] <= 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dElderRay_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0