#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1w trend filter.
# Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price.
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when: Alligator aligned bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND 1w close > 1w EMA21.
# Short when: Alligator aligned bearish (Lips < Teeth < Jaw) AND Bear Power > 0 AND 1w close < 1w EMA21.
# Uses ATR-based trailing stop: exit long if price < highest_since_entry - 2.0*ATR(20),
# exit short if price > lowest_since_entry + 2.0*ATR(20).
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).

name = "6h_WilliamsAlligator_ElderRay_1wTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w EMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Williams Alligator components on median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for Alligator (max shift 8), EMA13, ATR
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        
        # Williams Alligator alignment
        bullish_aligned = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_aligned = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray signals
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] > 0
        
        # 1w trend filter
        uptrend_1w = curr_close > ema_21_1w_aligned[i]
        downtrend_1w = curr_close < ema_21_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator + Bull Power > 0 + 1w uptrend
            if bullish_aligned and bull_strong and uptrend_1w:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Bearish Alligator + Bear Power > 0 + 1w downtrend
            elif bearish_aligned and bear_strong and downtrend_1w:
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
            
            # Exit conditions: ATR stoploss OR Alligator reversal OR Elder Ray weakening
            stop_price = highest_since_entry - 2.0 * curr_atr
            if (curr_close < stop_price or 
                not bullish_aligned or 
                bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Alligator reversal OR Elder Ray weakening
            stop_price = lowest_since_entry + 2.0 * curr_atr
            if (curr_close > stop_price or 
                not bearish_aligned or 
                bear_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals