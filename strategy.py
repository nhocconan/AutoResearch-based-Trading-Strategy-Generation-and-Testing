#!/usr/bin/env python3
name = "6h_1d_Momentum_Divergence_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1D close for momentum and volume analysis
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1D RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1D volume moving average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d = np.nan_to_num(vol_ratio_1d, nan=1.0)
    
    # Calculate 60-period price momentum (rate of change) on 1D
    mom_60 = np.zeros_like(close_1d)
    mom_60[60:] = (close_1d[60:] - close_1d[:-60]) / close_1d[:-60] * 100
    
    # Align 1D indicators to 6H timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    mom_60_aligned = align_htf_to_ltf(prices, df_1d, mom_60)
    
    # 6H price momentum (10-period ROC) for entry timing
    roc_10 = np.zeros_like(close)
    roc_10[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup
    start_idx = 70
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(mom_60_aligned[i]) or np.isnan(roc_10[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require above-average volume on 1D
        volume_filter = vol_ratio_1d_aligned[i] > 1.3
        
        if position == 0:
            # Long: Bullish momentum divergence - price making higher lows but RSI making lower lows
            # Simplified: Look for RSI oversold (<30) with improving momentum and volume support
            if (rsi_1d_aligned[i] < 30 and 
                mom_60_aligned[i] > -5 and  # Momentum not severely negative
                roc_10[i] > 0 and           # Short-term positive momentum
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bearish momentum divergence - price making lower highs but RSI making higher highs
            elif (rsi_1d_aligned[i] > 70 and 
                  mom_60_aligned[i] < 5 and   # Momentum not excessively positive
                  roc_10[i] < 0 and           # Short-term negative momentum
                  volume_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: momentum exhaustion or RSI extreme reversal
            if position == 1:
                # Exit long: RSI overbought or momentum turning negative
                if (rsi_1d_aligned[i] > 70) or (roc_10[i] < -0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI oversold or momentum turning positive
                if (rsi_1d_aligned[i] < 30) or (roc_10[i] > 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals