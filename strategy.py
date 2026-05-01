#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + TRIX confluence on 1w HTF trend filter.
# Long when: Alligator bullish (jaw < teeth < lips), Elder Bull Power > 0, TRIX rising, and 1w EMA50 uptrend.
# Short when: Alligator bearish (jaw > teeth > lips), Elder Bear Power < 0, TRIX falling, and 1w EMA50 downtrend.
# Uses ATR-based trailing stop: exit if price moves against position by 3.0*ATR(21) from favorable extreme.
# Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years) to avoid fee drag.
# Williams Alligator: Jaw=SMMA(13,8), Teeth=SMMA(8,5), Lips=SMMA(5,3)
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# TRIX: Triple EMA(12) of close, then % change
# 1w HTF EMA50 ensures alignment with major weekly trend to reduce whipsaw in ranging markets.
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn.

name = "1d_WilliamsAlligator_ElderRay_TRIX_1wEMA50_Confluence_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w EMA50 for HTF trend filter (loaded once before loop)
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
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # SMMA(13)
    teeth = smma(close, 8)   # SMMA(8)
    lips = smma(close, 5)    # SMMA(5)
    
    # Shift Alligator lines by 5, 3, 0 respectively (Alligator method)
    jaw_shifted = np.roll(jaw, 5)
    teeth_shifted = np.roll(teeth, 3)
    lips_shifted = lips  # no shift
    
    # Elder Ray: EMA(13) of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # TRIX: Triple EMA(12) of close, then 1-period percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = pd.Series(ema3).pct_change() * 100  # Convert to percentage
    trix_values = trix.values
    trix_rising = trix_values > np.roll(trix_values, 1)
    trix_falling = trix_values < np.roll(trix_values, 1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for Alligator (max period 13), EMA13, ATR21, TRIX
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(trix_values[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        
        # Williams Alligator conditions
        alligator_bullish = (jaw_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < lips_shifted[i])
        alligator_bearish = (jaw_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > lips_shifted[i])
        
        # Elder Ray conditions
        elder_bull = bull_power[i] > 0
        elder_bear = bear_power[i] < 0
        
        # TRIX conditions
        trix_up = trix_rising[i]
        trix_down = trix_falling[i]
        
        # 1w HTF trend filter
        weekly_uptrend = curr_close > ema_50_1w_aligned[i]
        weekly_downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish AND Elder Bull Power > 0 AND TRIX rising AND weekly uptrend
            if alligator_bullish and elder_bull and trix_up and weekly_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Alligator bearish AND Elder Bear Power < 0 AND TRIX falling AND weekly downtrend
            elif alligator_bearish and elder_bear and trix_down and weekly_downtrend:
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
            
            # Exit conditions: ATR trailing stoploss OR Alligator bearish OR Elder Bear Power < 0 OR TRIX falling OR weekly downtrend
            stop_price = highest_since_entry - 3.0 * curr_atr
            if (curr_close < stop_price or 
                alligator_bearish or 
                not elder_bull or 
                trix_down or 
                weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR trailing stoploss OR Alligator bullish OR Elder Bull Power > 0 OR TRIX rising OR weekly uptrend
            stop_price = lowest_since_entry + 3.0 * curr_atr
            if (curr_close > stop_price or 
                alligator_bullish or 
                not elder_bear or 
                trix_up or 
                weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals