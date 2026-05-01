#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray power + 1w trend filter.
# Long when: Alligator bullish alignment (jaw < teeth < lips) AND Elder Bear Power < 0 AND price > 1w EMA50.
# Short when: Alligator bearish alignment (jaw > teeth > lips) AND Elder Bull Power > 0 AND price < 1w EMA50.
# Uses ATR-based trailing stop: exit long if price < highest_since_entry - 2.5*ATR(21),
# exit short if price > lowest_since_entry + 2.5*ATR(21).
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid fee drag.
# Alligator smoothed with SMMA ( Wilder's smoothing ) using EMA as proxy.
# Williams Alligator shows market phases: sleeping (all lines intertwined), waking up (lines diverge),
# eating (trend strong, lines separated and ordered), sated (lines converge again).
# Elder Ray measures bull/bear power relative to EMA13 to assess trend strength behind price moves.
# Combining both gives high-conviction trend entries with built-in trend strength filter.

name = "6h_WilliamsAlligator_ElderRay_1wTrend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w EMA50 for trend filter (loaded once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 21-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA (using EMA as proxy)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for Alligator, Elder Ray, ATR, and 1w EMA
    start_idx = 150
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        
        # Alligator alignment
        bullish_alligator = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alligator = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray power
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator AND Bear Power negative (bulls in control despite bears) AND price > 1w EMA50
            if bullish_alligator and (bear_power_val < 0) and (curr_close > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Bearish Alligator AND Bull Power positive (bears in control despite bulls) AND price < 1w EMA50
            elif bearish_alligator and (bull_power_val > 0) and (curr_close < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Alligator sleep (jaw-teeth-lips converge) OR Elder Ray weakness
            stop_price = highest_since_entry - 2.5 * curr_atr
            alligator_sleeping = abs(jaw[i] - teeth[i]) < (curr_atr * 0.5) and abs(teeth[i] - lips[i]) < (curr_atr * 0.5)
            if curr_close < stop_price or alligator_sleeping or (bull_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Alligator sleep OR Elder Ray weakness
            stop_price = lowest_since_entry + 2.5 * curr_atr
            alligator_sleeping = abs(jaw[i] - teeth[i]) < (curr_atr * 0.5) and abs(teeth[i] - lips[i]) < (curr_atr * 0.5)
            if curr_close > stop_price or alligator_sleeping or (bear_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals