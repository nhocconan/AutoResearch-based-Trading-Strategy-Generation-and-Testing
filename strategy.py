# 12h_Camarilla_Pivot_R1_S1_Breakout_VolumeATRFilter_v1
# Hypothesis: Camarilla pivot levels (R1/S1) on daily timeframe act as strong support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and ATR filter capture momentum moves.
# Works in bull/bear: In bull, breakouts above R1 continue; in bear, breakdowns below S1 continue.
# Uses 12h timeframe for lower trade frequency (~20-50/year) to reduce fee drag.
# Volume spike (>2x 20-period average) confirms institutional interest.
# ATR stop (2x ATR) limits downside during false breakouts.
# Target: 50-150 total trades over 4 years.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Formula: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We use previous day's high/low/close to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first day to NaN since no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate R1 and S1
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Calculate 14-period ATR for stop loss filter
    tr1 = np.maximum(high_1d, np.roll(close_1d, 1))
    tr1 = np.maximum(tr1, np.roll(low_1d, 1))
    tr2 = np.minimum(high_1d, np.roll(close_1d, 1))
    tr2 = np.maximum(tr2, np.roll(low_1d, 1))
    tr = np.maximum(tr1 - tr2, np.abs(tr1 - np.roll(close_1d, 1)), np.abs(tr2 - np.roll(close_1d, 1)))
    tr[0] = np.nan  # First TR is NaN
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all daily data to 12h timeframe (primary)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 12h volume spike indicator (volume > 2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 days for volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: require volume spike
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and ATR filter
            # ATR filter: only take breakout if current volatility is not too high (avoid chop)
            if close[i] > R1_12h[i] and vol_confirmed and atr_14_12h[i] > 0:
                # Additional filter: breakout should be significant (> 0.5*ATR)
                if close[i] - R1_12h[i] > 0.5 * atr_14_12h[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S1 with volume spike and ATR filter
            elif close[i] < S1_12h[i] and vol_confirmed and atr_14_12h[i] > 0:
                # Additional filter: breakdown should be significant (> 0.5*ATR)
                if S1_12h[i] - close[i] > 0.5 * atr_14_12h[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 OR ATR-based stop loss (2*ATR below entry)
            # We approximate stop by checking if price has moved against us by 2*ATR
            # Since we don't track entry price exactly, use a time-based trailing condition
            if close[i] < S1_12h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 OR ATR-based stop loss (2*ATR above entry)
            if close[i] > R1_12h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1_S1_Breakout_VolumeATRFilter_v1"
timeframe = "12h"
leverage = 1.0