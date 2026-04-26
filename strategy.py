#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: 1d Camarilla R1/S1 breakout in direction of 1w EMA34 trend, confirmed by volume spike (>1.5x 20-bar MA). Designed for low frequency (target 10-30 trades/year) to minimize fee drag, works in bull/bear via weekly trend alignment. Uses discrete position sizing (0.25) to minimize churn.
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1w trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1w = close_val > ema_34_val
        bearish_1w = close_val < ema_34_val
        
        # Camarilla levels (based on previous day's OHLC)
        if i >= 1:
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            range_val = prev_high - prev_low
            
            # Camarilla R1 and S1 levels
            camarilla_r1 = prev_close + range_val * 1.1 / 12
            camarilla_s1 = prev_close - range_val * 1.1 / 12
        else:
            camarilla_r1 = high_val
            camarilla_s1 = low_val
        
        # Entry conditions
        long_entry = (close_val > camarilla_r1) and bullish_1w and vol_spike
        short_entry = (close_val < camarilla_s1) and bearish_1w and vol_spike
        
        # Exit conditions: opposite Camarilla level touch
        exit_long = (close_val < camarilla_s1) and position == 1
        exit_short = (close_val > camarilla_r1) and position == -1
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0