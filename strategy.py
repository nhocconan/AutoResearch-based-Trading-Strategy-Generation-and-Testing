#!/usr/bin/env python3
# 1d_1w_vwap_reversion_v1
# Strategy: Daily VWAP mean reversion with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Price reverts to daily VWAP with confluence of weekly trend alignment and volume spikes.
# Long when price crosses below daily VWAP with volume > 2x 20-day average and weekly VWAP trend up.
# Short when price crosses above daily VWAP with volume > 2x 20-day average and weekly VWAP trend down.
# Uses tight entry conditions to limit trades (~20-40/year) and avoid fee drag.
# Works in ranging markets (mean reversion) and trending markets (pullbacks to VWAP in trend direction).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_vwap_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative sums for VWAP calculation
    cum_vwap_num = np.nancumsum(vwap_numerator)
    cum_vwap_den = np.nancumsum(vwap_denominator)
    vwap = np.divide(cum_vwap_num, cum_vwap_den, out=np.zeros_like(cum_vwap_num), where=cum_vwap_den!=0)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly VWAP for trend filter
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap_num_1w = typical_price_1w * df_1w['volume']
    vwap_den_1w = df_1w['volume']
    cum_vwap_num_1w = np.nancumsum(vwap_num_1w)
    cum_vwap_den_1w = np.nancumsum(vwap_den_1w)
    vwap_1w = np.divide(cum_vwap_num_1w, cum_vwap_den_1w, out=np.zeros_like(cum_vwap_num_1w), where=cum_vwap_den_1w!=0)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Volume confirmation: current volume > 2x 20-day average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after VWAP warmup
        # Skip if any required data is invalid
        if (vwap_den_1w == 0 and i >= len(vwap_1w)) or np.isnan(vwap[i]) or np.isnan(vwap_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # VWAP cross signals
        vwap_cross_down = close[i] < vwap[i] and close[i-1] >= vwap[i-1]
        vwap_cross_up = close[i] > vwap[i] and close[i-1] <= vwap[i-1]
        
        # Weekly VWAP trend filter
        weekly_uptrend = vwap_1w_aligned[i] > vwap_1w_aligned[i-1] if i > 0 else True
        weekly_downtrend = vwap_1w_aligned[i] < vwap_1w_aligned[i-1] if i > 0 else True
        
        # Entry logic: VWAP cross + volume + weekly trend alignment
        if vwap_cross_down and vol_confirm[i] and weekly_uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif vwap_cross_up and vol_confirm[i] and weekly_downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite VWAP cross with volume confirmation
        elif position == 1 and vwap_cross_up and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and vwap_cross_down and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals