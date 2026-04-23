#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe strategy using 1w Camarilla pivot levels (H3/L3) with 1w EMA34 trend filter and volume confirmation (>1.5x average).
- Uses 1w for signal direction (Camarilla H3/L3 breakout) and 1w for trend filter (EMA34)
- Volume confirmation reduces false breakouts
- Session filter: 08-20 UTC to avoid low-liquidity periods
- Position size: 0.25 (discrete level to minimize fee churn)
- Target: 7-25 trades/year (30-100 over 4 years) to avoid fee drag
- Works in bull/bear via trend filter and volume confirmation
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 1.5x 24-period average (strict for 1d)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1w Camarilla pivot levels (H3, L3, H4, L4)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Calculate range
    range_1w = high_1w - low_1w
    # Camarilla levels
    h3_1w = pp_1w + range_1w * 1.1 / 4
    l3_1w = pp_1w - range_1w * 1.1 / 4
    h4_1w = pp_1w + range_1w * 1.1 / 2
    l4_1w = pp_1w - range_1w * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe (use prior completed 1w bar)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    h4_1w_aligned = align_htf_to_ltf(prices, df_1w, h4_1w)
    l4_1w_aligned = align_htf_to_ltf(prices, df_1w, l4_1w)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)  # EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(h3_1w_aligned[i]) or
            np.isnan(l3_1w_aligned[i]) or
            np.isnan(h4_1w_aligned[i]) or
            np.isnan(l4_1w_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up_h3 = close[i] > h3_1w_aligned[i-1]  # Close above prior 1w H3
        breakout_down_l3 = close[i] < l3_1w_aligned[i-1]  # Close below prior 1w L3
        
        if position == 0:
            # Long: 1w Camarilla H3 breakout up AND price > 1w EMA34 AND volume confirmation AND in session
            if breakout_up_h3 and volume_confirm and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: 1w Camarilla L3 breakout down AND price < 1w EMA34 AND volume confirmation AND in session
            elif breakout_down_l3 and volume_confirm and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1w Camarilla L4 break down OR price < 1w EMA34 (trend flip)
            if close[i] < l4_1w_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1w Camarilla H4 break up OR price > 1w EMA34 (trend flip)
            if close[i] > h4_1w_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_VolumeSpike_Session"
timeframe = "1d"
leverage = 1.0