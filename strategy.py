#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Pivot_R1S1_Breakout_VolumeATRFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Pivot calculation (same timeframe, no alignment needed)
    high_1d = high
    low_1d = low
    close_1d = close
    
    # Calculate Pivot, R1, S1 on 1d timeframe
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 50-period EMA on weekly for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(atr[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        # Volatility filter: require ATR > 0.5 * 50-period ATR average
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > (0.5 * atr_ma[i]) if not np.isnan(atr_ma[i]) else True
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume, volatility, and uptrend
            if close[i] > r1_1d[i] and vol_confirm and vol_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume, volatility, and downtrend
            elif close[i] < s1_1d[i] and vol_confirm and vol_filter and downtrend:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below S1 (reversal) or volatility drops
            if close[i] < s1_1d[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R1 (reversal) or volatility drops
            if close[i] > r1_1d[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals