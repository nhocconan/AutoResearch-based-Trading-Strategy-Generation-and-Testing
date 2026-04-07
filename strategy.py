#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with Elder Ray Bull/Bear power, filtered by weekly trend
# Uses Alligator's jaw/teeth/lips for trend direction and Elder Ray for momentum strength
# Weekly trend filter ensures alignment with higher timeframe bias
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag
# Works in bull/bear markets via trend filter and volatility-based position sizing

name = "6w_alligator_elder_ray_weekly_trend_v1"
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
    
    # 1-day data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Williams Alligator (13,8,5) - smoothed with SMMA
    close_1d = df_1d['close'].values
    # Jaw (13-period, smoothed 8 bars)
    jaw_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean()
    jaw = jaw_13.shift(8).values  # shift 8 bars forward
    # Teeth (8-period, smoothed 5 bars)
    teeth_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean()
    teeth = teeth_8.shift(5).values  # shift 5 bars forward
    # Lips (5-period, smoothed 3 bars)
    lips_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean()
    lips = lips_5.shift(3).values  # shift 3 bars forward
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    bull_power = high - ema_13_aligned
    bear_power = low - ema_13_aligned
    
    # Weekly trend filter: EMA(21) on weekly close
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # ATR for position sizing (based on 6h ATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals: aligned when jaw > teeth > lips (down) or jaw < teeth < lips (up)
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Alligator sleeping (intertwined) - no trend
        sleeping = (abs(jaw_val - teeth_val) < 0.001 * close[i]) and \
                   (abs(teeth_val - lips_val) < 0.001 * close[i]) and \
                   (abs(lips_val - jaw_val) < 0.001 * close[i])
        
        # Alligator awake with direction
        up_trend = jaw_val < teeth_val < lips_val
        down_trend = jaw_val > teeth_val > lips_val
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Elder Ray confirmation: strong bull/bear power
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-20):i+1])
        strong_bear = bear_power[i] < 0 and abs(bear_power[i]) > np.mean(abs(bear_power[max(0, i-20):i+1]))
        
        if position == 1:  # long position
            # Exit: Alligator reverses or weekly trend changes
            if down_trend or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator reverses or weekly trend changes
            if up_trend or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Enter long: Alligator up + weekly uptrend + strong bull power
            if up_trend and weekly_uptrend and strong_bull:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator down + weekly downtrend + strong bear power
            elif down_trend and weekly_downtrend and strong_bear:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals