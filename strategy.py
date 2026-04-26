#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Filter_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter (price > weekly EMA34) and volume confirmation.
Only trade breakouts in direction of weekly trend to avoid whipsaws. Uses R1/S1 as primary intraday support/resistance.
Long when price breaks above R1 with volume spike in weekly uptrend.
Short when price breaks below S1 with volume spike in weekly downtrend.
Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).
Designed for BTC/ETH - Camarilla pivots work in ranging markets, weekly trend filter adapts to bull/bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels on 1d
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low),
    # R2 = close + 0.5*(high-low), R1 = close + 0.25*(high-low),
    # S1 = close - 0.25*(high-low), S2 = close - 0.5*(high-low),
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    daily_range = df_1d['high'] - df_1d['low']
    camarilla_r1 = df_1d['close'] + 0.25 * daily_range
    camarilla_s1 = df_1d['close'] - 0.25 * daily_range
    
    # Align Camarilla levels to 1d timeframe (already aligned as we use 1d data)
    camarilla_r1_aligned = camarilla_r1.values
    camarilla_s1_aligned = camarilla_s1.values
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for HTF trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    htf_trend = np.where(close > ema_34_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for weekly EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (stricter threshold)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Breakout conditions
        breakout_above_r1 = close[i] > camarilla_r1_aligned[i]
        breakout_below_s1 = close[i] < camarilla_s1_aligned[i]
        
        if htf_trend[i] == 1:  # Weekly uptrend
            # Long signal: breakout above R1 with volume spike
            if breakout_above_r1 and volume_spike:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long: breakout below S1 (reversal signal)
            elif breakout_below_s1:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Weekly downtrend
            # Short signal: breakout below S1 with volume spike
            if breakout_below_s1 and volume_spike:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short: breakout above R1 (reversal signal)
            elif breakout_above_r1:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0