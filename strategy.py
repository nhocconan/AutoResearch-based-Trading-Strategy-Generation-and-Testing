#!/usr/bin/env python3
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
    
    # Get daily data for price channels and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channel (20-day)
    upper_1d = np.full_like(high_1d, np.nan)
    lower_1d = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-20:i])
        lower_1d[i] = np.min(low_1d[i-20:i])
    
    # Calculate daily ATR (14-day) for volatility filter
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly SMA(50) for trend filter
    if len(close_1w) >= 50:
        sma_1w = np.full_like(close_1w, np.nan)
        for i in range(50, len(close_1w)):
            sma_1w[i] = np.mean(close_1w[i-50:i])
    else:
        sma_1w = np.full_like(close_1w, np.nan)
    
    # Align all 1d data to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_12h = align_htf_to_ltf(prices, df_1d, lower_1d)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    sma_1w_12h = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Volume confirmation: volume > 1.8x 24-period average (balanced for 12h)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24  # 24 * 12h = 12 days
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 50, 24) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(sma_1w_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Trend filter: price above weekly SMA(50) = bullish bias
        bullish_bias = close[i] > sma_1w_12h[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume in bullish bias
            if close[i] > upper_12h[i] and vol_confirm and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume in bearish bias
            elif close[i] < lower_12h[i] and vol_confirm and not bullish_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR volatility drops (low vol = range)
            if close[i] < lower_12h[i] or atr_12h[i] < np.nanmedian(atr_12h[max(0, i-50):i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR volatility drops
            if close[i] > upper_12h[i] or atr_12h[i] < np.nanmedian(atr_12h[max(0, i-50):i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0