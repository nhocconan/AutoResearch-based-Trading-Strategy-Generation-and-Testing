#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray + Weekly Trend Filter
# Long when: price > Alligator Jaw, Bull Power > 0, and weekly close > weekly EMA50
# Short when: price < Alligator Jaw, Bear Power < 0, and weekly close < weekly EMA50
# Exit when price crosses back inside Alligator jaws or weekly trend changes
# Stoploss at 2.5 * ATR(21)
# Position size: 0.25 (25% of capital)
# Uses Williams Alligator (13,8,5 SMAs with shifts) and Elder Ray (EMA13 power)
# Target: 50-150 total trades over 4 years (12-38/year)

name = "6h_alligator_elder_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3)
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = close_s.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = close_s.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray Power: EMA13
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # ATR(21) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    # Weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_weekly_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price back inside Alligator jaws or weekly trend changes
            elif (close[i] < jaw[i] or close[i] > lips[i]) or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price back inside Alligator jaws or weekly trend changes
            elif (close[i] > jaw[i] or close[i] < lips[i]) or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Alligator alignment, Elder Ray power, and weekly trend
            # Bullish: price above Jaw, Bull Power positive, weekly uptrend
            bullish_setup = (close[i] > jaw[i] and 
                           bull_power[i] > 0 and 
                           close[i] > ema50_weekly_aligned[i])
            
            # Bearish: price below Jaw, Bear Power negative, weekly downtrend
            bearish_setup = (close[i] < jaw[i] and 
                           bear_power[i] < 0 and 
                           close[i] < ema50_weekly_aligned[i])
            
            if bullish_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals