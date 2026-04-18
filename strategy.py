#!/usr/bin/env python3
"""
1h_4hTrend_1dVolSpike_Entry
1h strategy using 4h trend filter and 1d volume spike for entry timing.
- Long: 4h EMA21 > EMA50 (uptrend) + 1h close > 1h VWAP + 1d volume > 1.5x 20d avg volume
- Short: 4h EMA21 < EMA50 (downtrend) + 1h close < 1h VWAP + 1d volume > 1.5x 20d avg volume
- Exit: Opposite trend on 4h or volume spike failure
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Uses 4h for trend direction, 1d for volume regime, 1h for precise entry timing
Works in bull trends (follow 4h uptrend) and bear trends (follow 4h downtrend)
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA21 and EMA50 for trend
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d 20-period volume average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for 4h EMA50 and 1d volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions from 4h
        uptrend_4h = ema_21_4h_aligned[i] > ema_50_4h_aligned[i]
        downtrend_4h = ema_21_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Volume spike condition from 1d
        vol_spike = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # 1h price relative to VWAP for entry timing
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]
        
        if position == 0:
            # Long: 4h uptrend + volume spike + price above VWAP
            if uptrend_4h and vol_spike and above_vwap:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + volume spike + price below VWAP
            elif downtrend_4h and vol_spike and below_vwap:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h trend turns down OR volume spike fails
            if not uptrend_4h or not vol_spike:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend turns up OR volume spike fails
            if not downtrend_4h or not vol_spike:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hTrend_1dVolSpike_Entry"
timeframe = "1h"
leverage = 1.0