#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + 1d Trend Filter + Volume Spike
# Long when Alligator jaws (13-period smoothed median) turns up, price > teeth (8-period), and 1d EMA34 rising + volume > 2x 20-period average
# Short when jaws turn down, price < teeth, and 1d EMA34 falling + volume > 2x 20-period average
# Williams Alligator is effective in ranging and trending markets, with jaws/teeth/lips providing dynamic support/resistance
# Uses ATR(14) trailing stop (2.5x) for risk control
# Discrete position sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) on 12h

name = "12h_WilliamsAlligator_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams Alligator on 12h timeframe
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaws: 13-period SMMA smoothed 8 bars ahead
    jaws_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws_raw, 8)  # Smoothed 8 bars ahead
    jaws[:8] = np.nan  # First 8 values invalid due to roll
    
    # Teeth: 8-period SMMA smoothed 5 bars ahead
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # Smoothed 5 bars ahead
    teeth[:5] = np.nan  # First 5 values invalid due to roll
    
    # Lips: 5-period SMMA smoothed 3 bars ahead
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # Smoothed 3 bars ahead
    lips[:3] = np.nan  # First 3 values invalid due to roll
    
    # Align Alligator components to 12h timeframe (wait for 12h bar to close)
    jaws_aligned = align_htf_to_ltf(prices, prices, jaws)  # Self-align since calculated on LTf
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Jaws turning up (jaws > previous jaws) AND price > teeth AND 1d EMA34 rising AND volume confirmation
            if (jaws_aligned[i] > jaws_aligned[i-1] and 
                close[i] > teeth_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Jaws turning down (jaws < previous jaws) AND price < teeth AND 1d EMA34 falling AND volume confirmation
            elif (jaws_aligned[i] < jaws_aligned[i-1] and 
                  close[i] < teeth_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals