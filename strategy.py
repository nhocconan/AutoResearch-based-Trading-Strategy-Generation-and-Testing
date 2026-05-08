#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Equilibrium_Point_Rebound_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and equilibrium point
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous 1d bar's equilibrium point (EP) and ATR
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Equilibrium point = (H + L + C) / 3
    EP = (prev_high + prev_low + prev_close) / 3.0
    
    # 1d ATR(14)
    high_low = prev_high - prev_low
    high_close = np.abs(prev_high - np.roll(prev_close, 1))
    low_close = np.abs(prev_low - np.roll(prev_close, 1))
    high_close[0] = prev_high[0] - prev_close[0]
    low_close[0] = prev_low[0] - prev_close[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all to 4h timeframe
    EP_aligned = align_htf_to_ltf(prices, df_1d, EP)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(EP_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid choppy markets)
        atr_ma50 = pd.Series(atr14_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr14_aligned[i] > atr_ma50[i] if not np.isnan(atr_ma50[i]) else True
        
        if position == 0:
            # Long: Price rebounds from below EP to above EP, with trend and volume
            long_cond = (close[i] > EP_aligned[i] and 
                        close[i-1] <= EP_aligned[i-1] and
                        close[i] > ema34_aligned[i] and
                        volume_spike[i] and
                        vol_filter)
            
            # Short: Price rejects from above EP to below EP, with trend and volume
            short_cond = (close[i] < EP_aligned[i] and 
                         close[i-1] >= EP_aligned[i-1] and
                         close[i] < ema34_aligned[i] and
                         volume_spike[i] and
                         vol_filter)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below EP OR crosses below EMA34
            if close[i] < EP_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above EP OR crosses above EMA34
            if close[i] > EP_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals