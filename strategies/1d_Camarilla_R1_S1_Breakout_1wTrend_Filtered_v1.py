#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Filtered_v1
Hypothesis: Daily Camarilla R1/S1 breakout in direction of weekly trend (EMA34) with volume confirmation.
Weekly EMA34 defines structural trend; breakouts aligned with it have higher follow-through.
Volume spike confirms institutional participation. Discrete sizing (0.25) limits fee drag.
Target: 75-125 total trades over 4 years (19-31/year) by requiring HTF alignment, breakout, and volume.
Works in bull/bear: weekly EMA34 adapts to regime; volume filter avoids false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    camarilla_r1 = close + 1.1 * (high - low) / 12
    camarilla_s1 = close - 1.1 * (high - low) / 12
    
    # Align daily Camarilla to 1d timeframe (no shift needed as we use same timeframe)
    camarilla_r1_aligned = camarilla_r1  # already daily
    camarilla_s1_aligned = camarilla_s1  # already daily
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for weekly EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions (use current bar's close vs previous day's levels)
        breakout_above = close[i] > camarilla_r1_aligned[i-1]  # Previous day's R1
        breakout_below = close[i] < camarilla_s1_aligned[i-1]  # Previous day's S1
        
        if breakout_above and volume_spike:
            # Long signal: breakout above Camarilla R1 with volume, above weekly EMA34 (bullish bias)
            if close[i] > ema_34_1w_aligned[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            else:
                # Hold or flatten if not aligned with weekly trend
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = 0.0
                    position = 0
        elif breakout_below and volume_spike:
            # Short signal: breakout below Camarilla S1 with volume, below weekly EMA34 (bearish bias)
            if close[i] < ema_34_1w_aligned[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # Hold or flatten if not aligned with weekly trend
                if position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
                    position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Filtered_v1"
timeframe = "1d"
leverage = 1.0