#/usr/bin/env python3
name = "12h_Vortex_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for trend filter and volume spike
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d volume average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Vortex Indicator (14 periods) on 12h data
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.abs(high[0] - low[0])
    vm_minus[0] = np.abs(high[0] - low[0])
    
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values / \
              pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values / \
               pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(vi_plus[i]) or np.isnan(vi_minus[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) AND price above 1d EMA50 (uptrend) AND volume spike (>1.5x average)
            if vi_plus[i] > vi_minus[i] and close[i] > ema_1d_aligned[i] and volume[i] > (vol_ma_1d_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+ (bearish trend) AND price below 1d EMA50 (downtrend) AND volume spike
            elif vi_minus[i] > vi_plus[i] and close[i] < ema_1d_aligned[i] and volume[i] > (vol_ma_1d_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VI- > VI+ (trend reversal) OR price below 1d EMA50
            if vi_minus[i] > vi_plus[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: VI+ > VI- (trend reversal) OR price above 1d EMA50
            if vi_plus[i] > vi_minus[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals