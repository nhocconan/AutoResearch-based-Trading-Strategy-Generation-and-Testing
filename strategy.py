#!/usr/bin/env python3
# 6h_1w_1d_donchian_breakout_volume_confirm_v1
# Hypothesis: 6-hour Donchian channel breakouts filtered by weekly trend direction and daily volume confirmation.
# Long when price breaks above 20-period high with weekly uptrend (price > weekly EMA50) and daily volume > 1.5x 20-day average.
# Short when price breaks below 20-period low with weekly downtrend (price < weekly EMA50) and daily volume > 1.5x 20-day average.
# Uses weekly trend filter to avoid counter-trend trades and volume confirmation to avoid false breakouts.
# Designed for 15-30 trades/year on 6h to minimize fee dust while capturing strong momentum moves.
# Works in bull markets via long breakouts and bear markets via short breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_donchian_breakout_volume_confirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: EMA50 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    daily_volume = df_1d['volume'].values
    daily_vol_ma20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_ma20)
    daily_vol_aligned = align_htf_to_ltf(prices, df_1d, daily_volume)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure Donchian is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(weekly_ema50_aligned[i]) or np.isnan(daily_vol_ma20_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current daily volume > 1.5x 20-day average
        vol_surge = False
        if daily_vol_ma20_aligned[i] > 0:
            vol_surge = daily_vol_aligned[i] > 1.5 * daily_vol_ma20_aligned[i]
        
        # Weekly trend: price above/below weekly EMA50
        weekly_uptrend = close[i] > weekly_ema50_aligned[i]
        weekly_downtrend = close[i] < weekly_ema50_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low
            if close[i] < low_roll[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high
            if close[i] > high_roll[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above 20-period high with weekly uptrend and volume surge
            if close[i] > high_roll[i] and weekly_uptrend and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: break below 20-period low with weekly downtrend and volume surge
            elif close[i] < low_roll[i] and weekly_downtrend and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals