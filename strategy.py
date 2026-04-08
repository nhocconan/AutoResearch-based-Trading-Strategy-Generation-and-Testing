#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_trend_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-week data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate EMA21 on weekly close
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMA to daily timeframe (wait for weekly bar to close)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume confirmation on daily: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 1.5
    
    # Trend filter: price above/below weekly EMA21
    price_above_ema = close > ema_21_aligned
    price_below_ema = close < ema_21_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if EMA is not available
        if np.isnan(ema_21_aligned[i]):
            if position != 0:
                # Hold position until exit
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly EMA21
            if price_below_ema[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly EMA21
            if price_above_ema[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above EMA with volume confirmation
            if price_above_ema[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below EMA with volume confirmation
            elif price_below_ema[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals