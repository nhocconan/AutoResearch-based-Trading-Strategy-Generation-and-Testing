#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_cci_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate CCI(20) on 1d typical price
    tp_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    mad_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (tp_1d - sma_tp) / (0.015 * mad_tp)
    # Replace inf/nan from division by zero
    cci_1d = np.where(np.isfinite(cci_1d), cci_1d, 0.0)
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 6h volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(cci_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below -100 (overbought exit)
            if cci_1d_aligned[i] < -100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above 100 (oversold exit)
            if cci_1d_aligned[i] > 100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: CCI crosses above -100 from below, above 1w EMA50, with volume confirmation
            if (cci_1d_aligned[i] > -100 and 
                i > 0 and cci_1d_aligned[i-1] <= -100 and
                close[i] > ema50_1w_aligned[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses below 100 from above, below 1w EMA50, with volume confirmation
            elif (cci_1d_aligned[i] < 100 and 
                  i > 0 and cci_1d_aligned[i-1] >= 100 and
                  close[i] < ema50_1w_aligned[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals