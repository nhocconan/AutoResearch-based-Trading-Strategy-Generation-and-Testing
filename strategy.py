#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot with daily volume confirmation and weekly trend filter.
# Long when price breaks above R1 with volume > 1.5x daily average and weekly close above 20-week EMA.
# Short when price breaks below S1 with volume > 1.5x daily average and weekly close below 20-week EMA.
# Exit when price crosses back through the pivot point (PP).
# Uses Camarilla for intraday structure, volume for confirmation, weekly EMA for trend filter.
# Target: 15-30 trades/year per symbol.
name = "6h_Camarilla_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot and volume average
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels for previous day
    # PP = (H + L + C) / 3
    # R1 = PP + (H - L) * 1.1 / 12
    # S1 = PP - (H - L) * 1.1 / 12
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = pp + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    s1 = pp - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    # Shift by 1 to use previous day's levels
    pp = pp.shift(1)
    r1 = r1.shift(1)
    s1 = s1.shift(1)
    
    # Get 1d average volume for confirmation (20-day MA)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter (20-week EMA of close)
    df_1w = get_htf_data(prices, '1w')
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all HTF arrays to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        ema20w = ema20_1w_aligned[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = close[i] > ema20w  # Using current close vs weekly EMA
        weekly_downtrend = close[i] < ema20w
        
        if position == 0:
            # Long entry: break above R1 + volume spike + weekly uptrend
            if price > r1_val and vol > 1.5 * vol_ma and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + weekly downtrend
            elif price < s1_val and vol > 1.5 * vol_ma and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point
            if price < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if price > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals