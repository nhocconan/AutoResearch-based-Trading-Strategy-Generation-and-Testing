#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike
Hypothesis: 1h Camarilla R1/S1 breakout in direction of 4h EMA34 trend, confirmed by 1d volume spike (>2x 20-bar MA). Uses discrete sizing (0.20) and session filter (08-20 UTC) to reduce overtrading. Designed for 15-37 trades/year on BTC/ETH in both bull and bear markets via trend alignment and volume confirmation.
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
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume MA20 for spike confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Volume spike: current 1d volume > 2x 20-day MA
    volume_spike_1d = vol_1d > (vol_ma_20_1d_aligned * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20  # Position size
    
    # Warmup: max of calculations
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0  # Force flat outside session
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_34_val = ema_34_4h_aligned[i]
        vol_spike = volume_spike_1d[i]
        
        # Determine 4h trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_4h = close_val > ema_34_val
        bearish_4h = close_val < ema_34_val
        
        # Camarilla levels (based on previous day's range)
        if i >= 24:  # Need at least 24 hours of 1h data for prior day
            # Prior day: 24h ago to now (excl current bar)
            prior_day_high = np.max(high[i-24:i])
            prior_day_low = np.min(low[i-24:i])
            prior_day_close = close[i-24]  # Close 24h ago
            
            # Camarilla R1, S1, R3, S3
            rang = prior_day_high - prior_day_low
            r1 = prior_day_close + rang * 1.1 / 12
            s1 = prior_day_close - rang * 1.1 / 12
            r3 = prior_day_close + rang * 1.1 / 4
            s3 = prior_day_close - rang * 1.1 / 4
        else:
            # Not enough data for Camarilla calculation
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Entry conditions
        long_entry = (close_val > r1) and bullish_4h and vol_spike
        short_entry = (close_val < s1) and bearish_4h and vol_spike
        
        # Exit conditions: reverse signal or opposite Camarilla level
        exit_long = (close_val < s3) or (not bullish_4h) or (not vol_spike)
        exit_short = (close_val > r3) or (not bearish_4h) or (not vol_spike)
        
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0