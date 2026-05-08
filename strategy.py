#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_Power_1dTrend"
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
    
    # Get 1d data once for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Elder Ray (Bull/Bear Power)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_1d
    bear_power = low - ema13_1d
    
    # Trend filter: EMA34 on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = (close_1d > ema34_1d).astype(float)
    trend_dn = (close_1d < ema34_1d).astype(float)
    
    # Align to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    trend_up_6h = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_dn_6h = align_htf_to_ltf(prices, df_1d, trend_dn)
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(trend_up_6h[i]) or np.isnan(trend_dn_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power positive, bear power negative, uptrend, volume filter
            long_cond = (bull_power_6h[i] > 0 and bear_power_6h[i] < 0 and 
                         trend_up_6h[i] > 0.5 and vol_filter[i])
            
            # Short: Bear power negative, bull power negative, downtrend, volume filter
            short_cond = (bear_power_6h[i] < 0 and bull_power_6h[i] < 0 and 
                          trend_dn_6h[i] > 0.5 and vol_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear power turns positive (bulls losing control)
            if bear_power_6h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull power turns positive (bears losing control)
            if bull_power_6h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter on 6h timeframe.
# Enters long when bull power > 0, bear power < 0 (bulls in control), 1d uptrend, and volume filter.
# Enters short when bear power < 0, bull power < 0 (bears in control), 1d downtrend, and volume filter.
# Exits when the opposing power turns positive (loss of control).
# Uses volume filter to avoid low-conviction moves. Targets 15-25 trades/year on 6h.
# Works in bull markets (follow bull power) and bear markets (follow bear power).
# Elder Ray measures buying/selling pressure relative to EMA13, effective in trending markets.