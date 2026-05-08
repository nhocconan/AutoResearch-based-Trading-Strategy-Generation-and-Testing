#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (bull/bear power) with weekly trend filter and volume confirmation
# Bull Power = High - EMA13 (13-period EMA of close)
# Bear Power = Low - EMA13
# Long when Bull Power > 0 and Bear Power rising (momentum) and weekly EMA34 uptrend and volume spike
# Short when Bear Power < 0 and Bull Power falling and weekly EMA34 downtrend and volume spike
# Weekly trend filter prevents counter-trend trades in strong trends
# Volume spike confirms institutional participation; avoids choppy false signals
# Elder Ray captures momentum shifts early; weekly filter improves robustness in bull/bear markets
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_ElderRay_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate EMA(13) for Elder Ray on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    
    # Momentum: slope of Bear Power (for short) and Bull Power (for long)
    # Using 3-period change as proxy for slope
    bull_power_mom = np.diff(bull_power, prepend=bull_power[0])
    bear_power_mom = np.diff(bear_power, prepend=bear_power[0])
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend = ema34_1w_aligned[i]
        curr_close = close[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_bull_mom = bull_power_mom[i]
        curr_bear_mom = bear_power_mom[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 AND Bull Power momentum positive AND weekly uptrend AND volume spike
            if curr_bull > 0 and curr_bull_mom > 0 and curr_close > weekly_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 AND Bear Power momentum negative AND weekly downtrend AND volume spike
            elif curr_bear < 0 and curr_bear_mom < 0 and curr_close < weekly_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or weekly trend turns down
            if curr_bull <= 0 or curr_close < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or weekly trend turns up
            if curr_bear >= 0 or curr_close > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals