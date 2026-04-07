#!/usr/bin/env python3
"""
1h_volume_momentum_4h1d_trend_v1
Hypothesis: On 1h timeframe, use volume-weighted momentum (VWM) for entry timing, with 4h trend filter and 1d regime filter.
VWM = (close - open) * volume. Enter long when VWM > 0 and 4h EMA50 > EMA200 and 1d ATR ratio < 0.8 (low volatility regime).
Enter short when VWM < 0 and 4h EMA50 < EMA200 and 1d ATR ratio < 0.8.
Exit when momentum reverses or volatility increases (ATR ratio > 1.2).
Targets 15-37 trades/year by combining momentum timing with trend/regime filters to reduce false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_momentum_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume-weighted momentum: (close - open) * volume
    vwm = (close - open_) * volume
    
    # 4h trend filter: EMA50 and EMA200
    close_s = pd.Series(close)
    ema50_4h_data = get_htf_data(prices, '4h')
    ema200_4h_data = get_htf_data(prices, '4h')
    
    if len(ema50_4h_data) < 50 or len(ema200_4h_data) < 200:
        return np.zeros(n)
    
    close_4h = ema50_4h_data['close'].values
    close_s_4h = pd.Series(close_4h)
    ema50_4h = close_s_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_4h = close_s_4h.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    ema50_4h_aligned = align_htf_to_ltf(prices, ema50_4h_data, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, ema200_4h_data, ema200_4h)
    
    # 1d regime filter: ATR ratio (current ATR / 20-period ATR) to detect low volatility
    atr_data = get_htf_data(prices, '1d')
    if len(atr_data) < 30:
        return np.zeros(n)
    
    high_1d = atr_data['high'].values
    low_1d = atr_data['low'].values
    close_1d = atr_data['close'].values
    
    # Calculate True Range and ATR
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d
    
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, atr_data, atr_ratio_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if required data not available or outside session
        if (np.isnan(vwm[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or
            np.isnan(atr_ratio_1d_aligned[i]) or
            not session_mask[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h
        trend_up = ema50_4h_aligned[i] > ema200_4h_aligned[i]
        trend_down = ema50_4h_aligned[i] < ema200_4h_aligned[i]
        
        # Regime filter: low volatility (ATR ratio < 0.8)
        low_vol = atr_ratio_1d_aligned[i] < 0.8
        high_vol_exit = atr_ratio_1d_aligned[i] > 1.2
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on momentum reversal (VWM <= 0)
            if vwm[i] <= 0:
                exit_long = True
            # Exit on trend reversal
            elif ema50_4h_aligned[i] < ema200_4h_aligned[i]:
                exit_long = True
            # Exit on high volatility
            elif high_vol_exit:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on momentum reversal (VWM >= 0)
            if vwm[i] >= 0:
                exit_short = True
            # Exit on trend reversal
            elif ema50_4h_aligned[i] > ema200_4h_aligned[i]:
                exit_short = True
            # Exit on high volatility
            elif high_vol_exit:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: positive VWM, uptrend, low volatility
            long_entry = (vwm[i] > 0) and trend_up and low_vol
            
            # Short entry: negative VWM, downtrend, low volatility
            short_entry = (vwm[i] < 0) and trend_down and low_vol
            
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
    
    return signals