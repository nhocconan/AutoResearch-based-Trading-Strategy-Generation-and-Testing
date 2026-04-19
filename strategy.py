#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_CCI_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for CCI and trend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h CCI(20)
    tp_12h = (high_12h + low_12h + close_12h) / 3.0
    sma_tp = pd.Series(tp_12h).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp_12h).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-10, mad)
    cci_12h = (tp_12h - sma_tp) / (0.015 * mad)
    
    # Align CCI to 6h
    cci_12h_aligned = align_htf_to_ltf(prices, df_12h, cci_12h)
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(cci_12h_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        cci = cci_12h_aligned[i]
        ema_trend = ema_34_12h_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: CCI > -100 (oversold recovery) + price above EMA trend + volume
            if cci > -100 and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: CCI < 100 (overbought pullback) + price below EMA trend + volume
            elif cci < 100 and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: CCI < -100 (re-enter oversold) or price below EMA
            if cci < -100 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: CCI > 100 (re-enter overbought) or price above EMA
            if cci > 100 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals