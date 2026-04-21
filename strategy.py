#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 3-bar swing + 1d volume profile + 1w trend filter.
# Identifies short-term exhaustion moves against the weekly trend.
# In uptrend (price > 1w EMA50): go long on 3-bar downswing with high volume.
# In downtrend (price < 1w EMA50): go short on 3-bar upswing with high volume.
# Uses 1d volume spike to confirm institutional participation in the reversal.
# Target: 15-30 trades/year by requiring weekly trend alignment + 3-bar swing + volume spike.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load 1d for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_w = df_1w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume and its 20-period average
    vol_d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly and daily indicators to 6h
    ema50_w_aligned = align_htf_to_ltf(prices, df_1w, ema50_w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        vol_current = df_1d['volume'].iloc[i // 96] if i >= 96 else 0  # daily volume aligned to 6h (96 bars/day)
        if i // 96 >= len(vol_d):
            vol_current = 0
        
        # Weekly trend filter
        uptrend = prices['close'].iloc[i] > ema50_w_aligned[i]
        downtrend = prices['close'].iloc[i] < ema50_w_aligned[i]
        
        # 3-bar swing detection (requires 3 consecutive closes in same direction)
        if i >= 2:
            close1 = prices['close'].iloc[i-2]
            close2 = prices['close'].iloc[i-1]
            close3 = prices['close'].iloc[i]
            
            # 3-bar downswing: lower lows and lower closes
            downswing = (close2 < close1) and (close3 < close2)
            # 3-bar upswing: higher highs and higher closes
            upswing = (close2 > close1) and (close3 > close2)
        else:
            downswing = False
            upswing = False
        
        # Volume confirmation: current daily volume > 2.0x 20-day average
        volume_confirm = vol_current > 2.0 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Enter long in uptrend on 3-bar downswing with volume (pullback to buy)
            if uptrend and downswing and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short in downtrend on 3-bar upswing with volume (pullback to sell)
            elif downtrend and upswing and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reversal of the swing or volume drops
            exit_signal = False
            
            if position == 1:
                # Exit long: 3-bar upswing starts OR volume drops below average
                if upswing:
                    exit_signal = True
                elif vol_current < vol_ma_20_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: 3-bar downswing starts OR volume drops below average
                if downswing:
                    exit_signal = True
                elif vol_current < vol_ma_20_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_3BarSwing_1dVolume_1wTrend"
timeframe = "6h"
leverage = 1.0