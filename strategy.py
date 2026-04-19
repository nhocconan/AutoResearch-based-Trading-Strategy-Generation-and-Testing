#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_CCI_Volume_Trend_Filter"
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
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate CCI on 1h
    typical_price = (high + low + close) / 3
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical_price - tp_mean) / (0.015 * tp_mad)
    cci = cci.values
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1d volume average for volume filter
    vol_ma_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1h volume spike
    vol_ma_1h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_1h * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(cci[i]) or np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filters: price above 4h EMA34 and volume above 1d average
        uptrend_filter = close[i] > ema_34_4h_aligned[i] and volume[i] > vol_ma_1d_aligned[i]
        downtrend_filter = close[i] < ema_34_4h_aligned[i] and volume[i] > vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long when CCI < -100 (oversold) + uptrend filter + volume spike
            if cci[i] < -100 and uptrend_filter and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short when CCI > 100 (overbought) + downtrend filter + volume spike
            elif cci[i] > 100 and downtrend_filter and vol_confirm:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long position: exit when CCI > 0 (mean reversion) or trend fails
            if cci[i] > 0 or close[i] <= ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit when CCI < 0 (mean reversion) or trend fails
            if cci[i] < 0 or close[i] >= ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals