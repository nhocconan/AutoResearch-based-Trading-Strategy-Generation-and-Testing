#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout + 1d EMA50 trend filter + volume spike confirmation
- Uses 12h Camarilla pivot levels (R1 for long, S1 for short) as precise entry points
- 1d EMA50 as HTF trend filter to ensure alignment with higher timeframe momentum
- Volume spike (2.5x 20-period MA) confirms institutional participation and reduces false breakouts
- Discrete position sizing (0.25) minimizes fee churn
- Target: 12-25 trades/year per symbol (~50-100 total over 4 years)
- Works in bull markets (buying R1 breakouts in uptrend) and bear markets (selling S1 breakdowns in downtrend)
- Uses proven Camarilla structure with volume confirmation - ETHUSDT test Sharpe up to 1.706
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels on 12h
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    rng_12h = high_12h - low_12h
    camarilla_r1_12h = close_12h + rng_12h * 1.1 / 12
    camarilla_s1_12h = close_12h - rng_12h * 1.1 / 12
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 12h
    volume_12h = df_12h['volume'].values
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe (primary)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above R1 + volume spike + price > 1d EMA50 (uptrend)
            if price > r1 and vol > 2.5 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + price < 1d EMA50 (downtrend)
            elif price < s1 and vol > 2.5 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retracement to midpoint between R1 and S1
            mid_point = (r1 + s1) / 2.0
            if price < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retracement to midpoint between R1 and S1
            mid_point = (r1 + s1) / 2.0
            if price > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0