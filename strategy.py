#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator with Elder Ray force index and volume confirmation
    # Williams Alligator identifies trend direction via jaw/teeth/lips alignment.
    # Elder Ray measures bull/bear power to confirm trend strength.
    # Volume spike ensures institutional participation. Target: 12-37 trades/year.
    # Designed to work in both bull (trend following) and bear (counter-trend reversals) markets.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Alligator (13,8,5 SMAs of median price)
    df_1d = get_htf_data(prices, '1d')
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Load 1d data for Elder Ray (EMA13 of high/low)
    ema13_high = pd.Series(df_1d['high']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_low = pd.Series(df_1d['low']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_low
    bear_power = ema13_high - df_1d['low'].values
    
    # Align Elder Ray components to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + Bull Power > 0 + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                bull_power_aligned[i] > 0 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) + Bear Power > 0 + volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and
                  bear_power_aligned[i] > 0 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator alignment reverses or power fails
            if position == 1:
                if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and bull_power_aligned[i] > 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and bear_power_aligned[i] > 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dElderRay_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0