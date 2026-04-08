#!/usr/bin/env python3
# 6h_12h_1d_cci_trend_volume_v1
# Hypothesis: CCI(20) on 12h with 1d EMA trend filter and volume confirmation captures trend reversals in both bull/bear markets.
# Uses CCI to identify overbought/oversold conditions, EMA for trend direction, and volume to filter false signals.
# 6h timeframe reduces trade frequency vs lower TFs while maintaining responsiveness.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_cci_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for CCI calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate CCI(20) on 12h typical price
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tp_12h = (high_12h + low_12h + close_12h) / 3.0
    
    # CCI components
    tp_ma = pd.Series(tp_12h).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(tp_12h).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp_12h - tp_ma) / (0.015 * tp_mad)
    # Handle division by zero
    cci = np.where(tp_mad == 0, 0, cci)
    
    # Align CCI to 6h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_12h, cci)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Align to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 6h volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(cci_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 (overbought) OR price below EMA
            if cci_aligned[i] < 100 or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 (oversold) OR price above EMA
            if cci_aligned[i] > -100 or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: CCI crosses above -100 from below, price above EMA, with volume confirmation
            if cci_aligned[i] > -100 and cci_aligned[i-1] <= -100 and close[i] > ema50_1d_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI crosses below +100 from above, price below EMA, with volume confirmation
            elif cci_aligned[i] < 100 and cci_aligned[i-1] >= 100 and close[i] < ema50_1d_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals