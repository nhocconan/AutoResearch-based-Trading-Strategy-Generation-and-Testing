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
    
    # Load daily data for volatility and momentum signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period daily ADX for trend strength
    def calculate_adx(high, low, close, period=20):
        if len(high) < period + 1:
            return np.full(len(high), np.nan)
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        tr = np.concatenate([[high[0] - low[0]], tr])
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            plus_di[period-1] = 100 * np.mean(plus_dm[:period]) / atr[period-1] if atr[period-1] != 0 else 0
            minus_di[period-1] = 100 * np.mean(minus_dm[:period]) / atr[period-1] if atr[period-1] != 0 else 0
            
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_di[i] = 100 * ((plus_di[i-1] * (period-1) + plus_dm[i]) / period) / atr[i] if atr[i] != 0 else 0
                minus_di[i] = 100 * ((minus_di[i-1] * (period-1) + minus_dm[i]) / period) / atr[i] if atr[i] != 0 else 0
        
        # DX and ADX
        dx = np.zeros_like(tr)
        adx = np.full(len(tr), np.nan)
        
        for i in range(period, len(tr)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        if len(tr) >= 2 * period - 1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(tr)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_20 = calculate_adx(high_1d, low_1d, close_1d, 20)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_20)
    
    # Calculate 14-period daily ATR for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (20-period average on 6h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.28  # 28% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_6h[i]) or 
            np.isnan(atr_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.6% of price)
        if atr_6h[i] < 0.006 * close[i]:
            signals[i] = 0.0
            continue
        
        # Skip weak trend periods (ADX < 20)
        if adx_6h[i] < 20:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 1.8
        
        # Price momentum: 3-period ROC on 6h close
        roc_3 = 0
        if i >= 3:
            roc_3 = (close[i] - close[i-3]) / close[i-3] * 100
        
        if position == 0:
            # Long: Strong uptrend (ADX > 25) + upward momentum + volume confirmation
            if (adx_6h[i] > 25 and roc_3 > 0.3 and volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Strong downtrend (ADX > 25) + downward momentum + volume confirmation
            elif (adx_6h[i] > 25 and roc_3 < -0.3 and volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Trend weakening (ADX < 20) or momentum reversal
            if (adx_6h[i] < 20 or roc_3 < -0.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Trend weakening (ADX < 20) or momentum reversal
            if (adx_6h[i] < 20 or roc_3 > 0.2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ADX_Momentum_Volume_Filter"
timeframe = "6h"
leverage = 1.0