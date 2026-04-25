#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dEMA34_Filter
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA20 trend filter and 1d EMA34 regime filter.
Only long when price breaks above R1 in bull regime (price > 4h EMA20 AND price > 1d EMA34),
short when breaks below S1 in bear regime (price < 4h EMA20 AND price < 1d EMA34).
Session filter: 08-20 UTC to avoid low-liquidity hours. Discrete sizing 0.20.
Target: 15-35 trades/year by requiring confluence of 1h breakout + 4h trend + 1d regime.
Uses 4h/1d for signal direction, 1h only for entry timing to minimize fee drag.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for regime
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1h OHLC for Camarilla levels (use 1h data)
    # For 1h timeframe, we need to compute Camarilla from 1h OHLC
    # But we need to align it properly - compute on 1h then use as-is
    camarilla_r1_1h = close + (high - low) * 1.1 / 12
    camarilla_s1_1h = close - (high - low) * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 4h EMA20 (20) and 1d EMA34 (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_1h[i]) or np.isnan(camarilla_s1_1h[i])):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend regime and 1d regime
        # Bull regime: price > 4h EMA20 AND price > 1d EMA34
        # Bear regime: price < 4h EMA20 AND price < 1d EMA34
        if close[i] > ema_20_4h_aligned[i] and close[i] > ema_34_1d_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_20_4h_aligned[i] and close[i] < ema_34_1d_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades
        
        if position == 0:
            # Long setup: price breaks above R1 AND bull regime
            long_setup = (close[i] > camarilla_r1_1h[i]) and (regime == 'bull')
            
            # Short setup: price breaks below S1 AND bear regime
            short_setup = (close[i] < camarilla_s1_1h[i]) and (regime == 'bear')
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price breaks below S1 OR regime turns bearish/range
            if (close[i] < camarilla_s1_1h[i]) or (regime != 'bull'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price breaks above R1 OR regime turns bullish/range
            if (close[i] > camarilla_r1_1h[i]) or (regime != 'bear'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dEMA34_Filter"
timeframe = "1h"
leverage = 1.0