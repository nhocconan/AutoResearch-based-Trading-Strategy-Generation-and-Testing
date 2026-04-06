#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour price reversal at 12-hour volume-weighted VWAP extremes.
# Long when price crosses above VWAP with rising volume and bullish 12h momentum.
# Short when price crosses below VWAP with rising volume and bearish 12h momentum.
# Uses VWAP as dynamic support/resistance and volume confirmation to filter false breaks.
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and cost.

name = "4h_vwap_reversal_12h_mom_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_tpv, cum_vol, out=np.zeros_like(cum_tpv), where=cum_vol!=0)
    
    # 12h momentum: close vs close 3 periods ago (12h = 3 * 4h)
    close_series = pd.Series(close)
    mom_12h = close_series.diff(3).values  # positive = bullish momentum
    
    # Volume filter: current volume > 1.2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if momentum data not available
        if np.isnan(mom_12h[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.2
        
        # VWAP crossover conditions
        # Price above VWAP and rising
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]
        
        # Previous close relative to VWAP (for crossover detection)
        prev_close = close[i-1] if i > 0 else close[i]
        prev_above_vwap = prev_close > vwap[i-1] if i > 0 else above_vwap
        prev_below_vwap = prev_close < vwap[i-1] if i > 0 else below_vwap
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below VWAP or momentum turns bearish
            if (prev_above_vwap and below_vwap) or (mom_12h[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above VWAP or momentum turns bullish
            if (prev_below_vwap and above_vwap) or (mom_12h[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: cross above VWAP with bullish 12h momentum
                if (prev_below_vwap and above_vwap and mom_12h[i] > 0):
                    signals[i] = 0.25
                    position = 1
                # Short: cross below VWAP with bearish 12h momentum
                elif (prev_above_vwap and below_vwap and mom_12h[i] < 0):
                    signals[i] = -0.25
                    position = -1
    
    return signals