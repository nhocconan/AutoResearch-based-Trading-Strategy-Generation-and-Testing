#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d CCI(20) for overbought/oversold
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    sma_tp = pd.Series(typical_price_1d).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (typical_price_1d - sma_tp) / (0.015 * mad)
    
    # Align 1d CCI to 6h timeframe
    cci_6h_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 6h timeframe
    atr_6h_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h volume moving average (20-period) for volume filter
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        volume_series = pd.Series(volume)
        volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(cci_6h_aligned[i]) or np.isnan(atr_6h_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_6h_aligned[i] / close[i] < 0.003:
            signals[i] = 0.0
            continue
        
        # Skip low volume periods (volume < 60% of 20-period MA)
        if volume[i] < 0.6 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CCI < -100 (oversold) and price above prior close (bullish momentum)
            if cci_6h_aligned[i] < -100 and close[i] > close[i-1]:
                position = 1
                signals[i] = position_size
            # Short: CCI > 100 (overbought) and price below prior close (bearish momentum)
            elif cci_6h_aligned[i] > 100 and close[i] < close[i-1]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: CCI > -100 (exit oversold) or momentum fails
            if cci_6h_aligned[i] > -100 or close[i] < close[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: CCI < 100 (exit overbought) or momentum fails
            if cci_6h_aligned[i] < 100 or close[i] > close[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_CCI20_Momentum_Filter_v1"
timeframe = "6h"
leverage = 1.0