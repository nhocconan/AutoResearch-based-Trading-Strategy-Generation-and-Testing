#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_volatility_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend (EMA40)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # 1d data for volatility (ATR14)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 12h Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < Donchian low or trend fails
            if close[i] < donch_low[i] or close[i] < ema_40_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high or trend fails
            if close[i] > donch_high[i] or close[i] > ema_40_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volatility filter: avoid low volatility environments
            vol_filter_ok = atr_14_1d_aligned[i] > np.nanmedian(atr_14_1d_aligned[max(0, i-50):i+1])
            
            # Trend filter
            bullish = close[i] > ema_40_1w_aligned[i]
            bearish = close[i] < ema_40_1w_aligned[i]
            
            # Long: price > Donchian high + bullish trend + volume + volatility
            if (close[i] > donch_high[i] and 
                bullish and 
                vol_filter[i] and
                vol_filter_ok):
                position = 1
                signals[i] = 0.25
            # Short: price < Donchian low + bearish trend + volume + volatility
            elif (close[i] < donch_low[i] and 
                  bearish and 
                  vol_filter[i] and
                  vol_filter_ok):
                position = -1
                signals[i] = -0.25
    
    return signals