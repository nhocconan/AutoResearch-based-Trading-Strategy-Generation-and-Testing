#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator (13,8,5) with 1d Elder Ray (13) and volume confirmation
    # Williams Alligator identifies trend direction and strength via smoothed moving averages.
    # Elder Ray measures bull/bear power relative to EMA13 to confirm trend strength.
    # Combined with volume spike and session filter, this captures strong trends while avoiding whipsaws.
    # Designed for 12h timeframe to target 12-37 trades/year with low frequency and high conviction.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data for Williams Alligator (13,8,5 SMMA)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing
    def smma(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smma(close_12h, 13)  # Blue line (13-period)
    teeth = smma(close_12h, 8)  # Red line (8-period)
    lips = smma(close_12h, 5)   # Green line (5-period)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data for Elder Ray (13-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Net Power = Bull Power - Bear Power (positive = bullish, negative = bearish)
    net_power = bull_power - bear_power
    
    # Align Elder Ray to 1d timeframe
    net_power_aligned = align_htf_to_ltf(prices, df_1d, net_power)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(net_power_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator aligned: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
            # Elder Ray: Net Power > 0 = bullish, Net Power < 0 = bearish
            # Require volume confirmation
            
            # Long: Bullish alignment + positive Elder Ray + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                net_power_aligned[i] > 0 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + negative Elder Ray + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  net_power_aligned[i] < 0 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator reverses or Elder Ray contradicts position
            if position == 1:
                # Exit long if Alligator turns bearish or Elder Ray turns negative
                if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and net_power_aligned[i] > 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if Alligator turns bullish or Elder Ray turns positive
                if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and net_power_aligned[i] < 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dElderRay_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0