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
    
    # Get daily data for Donchian channel and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-day Donchian channel (using previous day's data)
    donch_high = np.full_like(high_1d, np.nan)
    donch_low = np.full_like(low_1d, np.nan)
    
    lookback = 20
    for i in range(lookback, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-lookback:i])
        donch_low[i] = np.min(low_1d[i-lookback:i])
    
    # Calculate 14-day ATR for volatility filter
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
    
    # Get weekly data for trend filter (EMA 34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all 1d data to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_1d, donch_low)
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1w_4h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 14, 34, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(atr_1d_4h[i]) or np.isnan(ema_1w_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above weekly EMA (bullish bias)
        bullish_bias = close[i] > ema_1w_4h[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and bullish bias
            if close[i] > donch_high_4h[i] and vol_confirm and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and bearish bias
            elif close[i] < donch_low_4h[i] and vol_confirm and not bullish_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR ATR expands (> 1.5x average)
            atr_ratio = atr_1d_4h[i] / np.nanmedian(atr_1d_4h[max(0, i-50):i]) if not np.isnan(atr_1d_4h[i]) and np.nanmedian(atr_1d_4h[max(0, i-50):i]) > 0 else 1.0
            if close[i] < donch_low_4h[i] or (atr_ratio > 1.5 and not bullish_bias):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR ATR expands (> 1.5x average) with bullish bias
            atr_ratio = atr_1d_4h[i] / np.nanmedian(atr_1d_4h[max(0, i-50):i]) if not np.isnan(atr_1d_4h[i]) and np.nanmedian(atr_1d_4h[max(0, i-50):i]) > 0 else 1.0
            if close[i] > donch_high_4h[i] or (atr_ratio > 1.5 and bullish_bias):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_EMA34Filter"
timeframe = "4h"
leverage = 1.0