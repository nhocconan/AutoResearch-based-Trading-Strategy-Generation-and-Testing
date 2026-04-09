#!/usr/bin/env python3
# 1h_mtf_volume_pullback_4h1d_v1
# Hypothesis: 1h strategy using 4h trend direction and 1d volume spike for entry timing.
# Long: 4h close > 4h EMA20 (uptrend) + 1h close pulls back to 1h VWAP + 1d volume > 1.5x 20-day average.
# Short: 4h close < 4h EMA20 (downtrend) + 1h close pulls back to 1h VWAP + 1d volume > 1.5x 20-day average.
# Exit: Position held for max 6 hours or opposite signal.
# Uses 4h for trend filter, 1d for volume regime, 1h for precise entry.
# Designed for low frequency (15-35 trades/year) to minimize fee drag in choppy 1h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_mtf_volume_pullback_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # 1h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    cum_tp_vol = np.nancumsum(tp_vol)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_tp_vol, cum_vol, out=np.full_like(cum_tp_vol, np.nan), where=cum_vol!=0)
    
    # 4h EMA20 for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d volume MA20 for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # already datetime64[ns]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(vwap[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(volume[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Pullback to VWAP: close within 0.5% of VWAP
        vwap_distance = abs(close[i] - vwap[i]) / vwap[i]
        near_vwap = vwap_distance < 0.005
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit: 6 hours max hold or reverse signal
            if bars_since_entry >= 6 or close[i] < ema_20_4h_aligned[i]:
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit: 6 hours max hold or reverse signal
            if bars_since_entry >= 6 or close[i] > ema_20_4h_aligned[i]:
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Reset timer
            bars_since_entry = 0
            
            # Long entry: 4h uptrend + pullback to VWAP + volume spike + in session
            if (close[i] > ema_20_4h_aligned[i] and    # 4h uptrend
                near_vwap and                          # Pullback to VWAP
                volume_confirmed and                   # Volume spike
                in_session[i]):                        # Active session
                position = 1
                signals[i] = 0.20
            # Short entry: 4h downtrend + pullback to VWAP + volume spike + in session
            elif (close[i] < ema_20_4h_aligned[i] and   # 4h downtrend
                  near_vwap and                         # Pullback to VWAP
                  volume_confirmed and                  # Volume spike
                  in_session[i]):                       # Active session
                position = -1
                signals[i] = -0.20
    
    return signals