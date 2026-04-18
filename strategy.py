#!/usr/bin/env python3
"""
4h_HTF_Confluence_Momentum_v1
Strategy: 4h EMA(34) trend with 1w EMA(34) filter + 1d volume spike + 1d price above 200 EMA.
Enters long when 4h EMA34 > price, 1w EMA34 up, 1d volume > 1.5x 20-period average, and 1d close > EMA200.
Enters short when 4h EMA34 < price, 1w EMA34 down, 1d volume spike, and 1d close < EMA200.
Uses 1w trend filter to avoid counter-trend trades in strong trends. Volume spike confirms institutional interest.
Designed for 4h timeframe: ~20-30 trades/year per symbol (80-120 total over 4 years).
Works in bull via uptrend filter, works in bear via downtrend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_prev = np.roll(ema_34_1w, 1)
    ema_34_1w_prev[0] = np.nan
    ema_34_1w_up = ema_34_1w > ema_34_1w_prev
    ema_34_1w_down = ema_34_1w < ema_34_1w_prev
    
    # Align weekly data to 4h timeframe
    ema_34_1w_up_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_up)
    ema_34_1w_down_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w_down)
    
    # Get daily data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 4h EMA34 for entry signal
    ema_34_4h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for daily EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_up_aligned[i]) or np.isnan(ema_34_1w_down_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_34_4h[i])):
            signals[i] = 0.0
            continue
        
        # 4h EMA34 vs price
        price_above_ema = close[i] > ema_34_4h[i]
        price_below_ema = close[i] < ema_34_4h[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Daily trend filter
        price_above_200ema = close[i] > ema_200_1d_aligned[i]
        price_below_200ema = close[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long: 4h price above EMA34, weekly uptrend, volume spike, daily above EMA200
            if price_above_ema and ema_34_1w_up_aligned[i] and vol_confirm and price_above_200ema:
                signals[i] = 0.25
                position = 1
            # Short: 4h price below EMA34, weekly downtrend, volume spike, daily below EMA200
            elif price_below_ema and ema_34_1w_down_aligned[i] and vol_confirm and price_below_200ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below EMA34 or weekly trend change to down
            if price_below_ema or ema_34_1w_down_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above EMA34 or weekly trend change to up
            if price_above_ema or ema_34_1w_up_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Confluence_Momentum_v1"
timeframe = "4h"
leverage = 1.0