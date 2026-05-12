# State your hypothesis:
# This strategy implements a 4h-based momentum reversal system using Elder Ray Index (Bull/Bear Power) with a 13-period EMA.
# It combines trend filtering via 12h EMA50 and volume confirmation to avoid false signals.
# The core idea: In strong trends, pullbacks to the EMA present continuation opportunities.
# Elder Ray Power > 0 indicates bullish momentum (bulls in control), < 0 indicates bearish momentum (bears in control).
# We go long when Bull Power turns positive after being negative (momentum shift up) in an uptrend.
# We go short when Bear Power turns negative after being positive (momentum shift down) in a downtrend.
# This captures momentum shifts at pullbacks, which works in both bull and bear markets by trading with the intermediate trend.
# Volume spike confirms institutional participation, reducing false signals.
# Exit when the power signal reverses, ensuring we don't overstay in reversals.
# Uses discrete position sizing (0.25) to minimize churn and keep trade frequency in the optimal range (est. 20-40/year).

#!/usr/bin/env python3
name = "4h_ElderRay_MomentumShift_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(13) for Elder Ray (same period used for EMA in Bull/Bear Power)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Components:
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure EMA13 and EMA50 have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power turns positive (was <=0) + 12h uptrend + volume spike
            if (bull_power[i] > 0 and bull_power[i-1] <= 0 and 
                close[i] > ema50_12h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power turns negative (was >=0) + 12h downtrend + volume spike
            elif (bear_power[i] < 0 and bear_power[i-1] >= 0 and 
                  close[i] < ema50_12h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power becomes negative (momentum shift down)
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power becomes positive (momentum shift up)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals