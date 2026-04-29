#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray volume-weighted trend filter
# Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Bull Power > 0 AND price > 1w EMA50
# Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND price < 1w EMA50
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee drag
# Target: 12-25 trades/year on 12h timeframe (~50-100 total over 4 years)
# Works in bull markets via Alligator uptrend + positive Elder Ray
# Works in bear markets via Alligator downtrend + negative Elder Ray
# Williams Alligator: SMAs of median price (HLC/3) with specific periods and shifts
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)

name = "12h_WilliamsAlligator_ElderRay_1wEMA50_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Williams Alligator on 12h data
    # Median price = (high + low + close) / 3
    median_price = (high + low + close) / 3.0
    
    # Alligator Jaw: SMA(13) of median, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Alligator Teeth: SMA(8) of median, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Alligator Lips: SMA(5) of median, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50, 50, 13, 8, 5)  # warmup for all indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_ema_13_1d = ema_13_1d_aligned[i]
        curr_atr = atr[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        
        # Skip if any indicator is not available
        if (np.isnan(curr_ema_50_1w) or np.isnan(curr_ema_13_1d) or np.isnan(curr_atr) or
            np.isnan(curr_jaw) or np.isnan(curr_teeth) or np.isnan(curr_lips)):
            signals[i] = 0.0
            continue
        
        # Calculate Elder Ray components
        bull_power = curr_high - curr_ema_13_1d
        bear_power = curr_low - curr_ema_13_1d
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR Alligator alignment breaks
            if curr_close < stop_price or not (curr_jaw < curr_teeth < curr_lips):
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR Alligator alignment breaks
            if curr_close > stop_price or not (curr_jaw > curr_teeth > curr_lips):
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Alligator bullish alignment AND Elder Bull Power > 0 AND price > 1w EMA50
            if (curr_jaw < curr_teeth < curr_lips and 
                bull_power > 0 and 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: Alligator bearish alignment AND Elder Bear Power < 0 AND price < 1w EMA50
            elif (curr_jaw > curr_teeth > curr_lips and 
                  bear_power < 0 and 
                  curr_close < curr_ema_50_1w):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals