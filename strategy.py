#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d Elder Ray and volume confirmation.
# Williams Alligator (SMAs with offsets) identifies trend direction and avoids chop.
# Elder Ray (13-period EMA vs 13-period high/low) confirms trend strength.
# Volume > 1.3x average confirms participation.
# Works in bull/bear as Alligator adapts to trend and filters false signals.
# Target: 20-35 trades/year per symbol (80-140 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    
    # 13-period EMA for Elder Ray
    ema_len = 13
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_13 = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # Williams Alligator on 4h: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: 1.3x average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, 13, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(ema_13_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator: aligned (jaws > teeth > lips = uptrend, reverse = downtrend)
        alligator_long = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])
        alligator_short = (jaws[i] < teeth[i]) and (teeth[i] < lips[i])
        
        # Elder Ray: bull power = high - EMA13, bear power = EMA13 - low
        bull_power = high[i] - ema_13_aligned[i]
        bear_power = ema_13_aligned[i] - low[i]
        # Require bull/bear power > 0 for confirmation
        elder_long = bull_power > 0
        elder_short = bear_power > 0
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: Alligator aligned up + Elder Ray bull + volume
            if (alligator_long and 
                elder_long and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Alligator aligned down + Elder Ray bear + volume
            elif (alligator_short and 
                  elder_short and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator reverses or Elder Ray turns bearish
            if not alligator_long or not elder_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator reverses or Elder Ray turns bullish
            if not alligator_short or not elder_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsAlligator_ElderRay_Volume_v1"
timeframe = "4h"
leverage = 1.0