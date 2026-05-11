#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike
# Hypothesis: Uses 1d Camarilla pivot levels (R3/S3) as key support/resistance. Enters long when price breaks above R3 with 1d uptrend (EMA34 rising) and volume spike; enters short when price breaks below S3 with 1d downtrend and volume spike. Exits when price returns to the 1d EMA34 or trend reverses. Works in bull markets by catching breakouts above resistance and in bear markets by catching breakdowns below support. Volume confirmation reduces false breakouts. 12h timeframe limits trades to avoid fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels, EMA34 trend, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla levels (R3, S3) ---
    # Formula: R3 = close + 1.1 * (high - low) * 1.1/2? Wait, standard Camarilla:
    # Actually: R4 = close + ((high-low) * 1.1/2), R3 = close + ((high-low) * 1.1/4)
    # But we want R3 and S3: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng / 4
    s3 = close_1d - 1.1 * rng / 4
    
    # --- 1d EMA34 for trend direction ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_slope = ema_34_1d - np.roll(ema_34_1d, 1)
    ema_34_1d_slope[0] = 0
    ema_34_1d_slope = pd.Series(ema_34_1d_slope).ewm(span=3, adjust=False, min_periods=1).mean().values  # smooth slope
    
    # --- 1d volume confirmation (volume > 20-period average) ---
    vol_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_slope)
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 (34) and smoothing (3)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(ema_34_1d_slope_aligned[i]) or
            np.isnan(vol_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1d EMA34 slope
        uptrend = ema_34_1d_slope_aligned[i] > 0
        downtrend = ema_34_1d_slope_aligned[i] < 0
        
        # Volume spike condition
        vol_spike = volume[i] > vol_20_1d_aligned[i] * 1.5  # 50% above average
        
        if position == 0:
            if uptrend and vol_spike:
                # Long: 1d uptrend + volume spike + price above R3
                if close[i] > r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_spike:
                # Short: 1d downtrend + volume spike + price below S3
                if close[i] < s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to EMA34 OR trend turns down
                if close[i] < ema_34_1d_aligned[i] or downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to EMA34 OR trend turns up
                if close[i] > ema_34_1d_aligned[i] or uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals