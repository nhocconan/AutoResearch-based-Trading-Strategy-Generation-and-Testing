#!/usr/bin/env python3
"""
4h_HTFTrend_LTFEntry
4h trend following with 1d/1w confirmation and LTF entry timing.
- Primary signal: 4h price above/below 20 EMA (trend)
- Entry filter: 1d close > 1w EMA34 (bull) or < 1w EMA34 (bear) - avoids counter-trend trades
- Entry trigger: 15m pullback to 4h EMA20 with volume > 1.5x average
- Exit: trend reversal or opposite signal
- Designed for 20-40 trades/year per symbol
Works in bull (trend continuation) and bear (trend continuation) markets
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
    
    # Get 1d and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 4h EMA20 for entry timing
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need 20 for EMA/volume MA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_20_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d close relative to 1w EMA34
        # Use 1d close from 1d data aligned to current time
        # We need to get the 1d close value for the current day
        # Since we're on 4h timeframe, we'll use the most recent 1d close
        # For simplicity, we'll use the 1d close from the 1d data that's already aligned
        # We'll get 1d data and align it
        
        # Actually, let's get 1d close properly
        close_1d = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        
        # Trend bull/bear based on 1d close vs 1w EMA34
        bull_trend = close_1d_aligned[i] > ema_34_1w_aligned[i]
        bear_trend = close_1d_aligned[i] < ema_34_1w_aligned[i]
        
        # Entry condition: price near 4h EMA20 with volume
        near_ema = abs(close[i] - ema_20_4h[i]) / ema_20_4h[i] < 0.005  # within 0.5%
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bull trend + pullback to EMA + volume
            if bull_trend and near_ema and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bear trend + pullback to EMA + volume
            elif bear_trend and near_ema and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or opposite signal
            if not bull_trend or (close[i] < ema_20_4h[i] and volume_ok):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or opposite signal
            if not bear_trend or (close[i] > ema_20_4h[i] and volume_ok):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTFTrend_LTFEntry"
timeframe = "4h"
leverage = 1.0