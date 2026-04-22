#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 120:  # need enough data for weekly and daily indicators
        return np.zeros(n)
    
    # Load 1w and 1d data once
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian(20) for trend
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_donchian = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily RSI(14) for mean reversion
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 6h
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1d, lower_donchian)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # 6h ATR(14) for position sizing and stop
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    tr1_6h = np.maximum(high_6h[1:] - low_6h[1:], np.abs(high_6h[1:] - close_6h[:-1]))
    tr2_6h = np.abs(low_6h[1:] - close_6h[:-1])
    tr_6h = np.concatenate([[np.inf], np.maximum(tr1_6h, tr2_6h)])
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(120, n):
        # Skip if any data is not ready
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(upper_donchian_aligned[i]) or 
            np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        atr_1w_val = atr_1w_aligned[i]
        upper_donch = upper_donchian_aligned[i]
        lower_donch = lower_donchian_aligned[i]
        rsi_val = rsi_1w_aligned[i]
        atr_6h_val = atr_6h[i]
        
        # Weekly volatility filter: only trade when weekly ATR is elevated
        vol_filter = atr_1w_val > np.nanpercentile(atr_1w_aligned[:i+1], 50)
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper + RSI < 50 (mean reversion in uptrend)
            if price > upper_donch and rsi_val < 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower + RSI > 50
            elif price < lower_donch and rsi_val > 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite Donchian break or RSI extreme
            exit_signal = False
            
            if position == 1:  # long position
                if price < lower_donch or rsi_val > 70:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > upper_donch or rsi_val < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchian_RSI_MeanReversion"
timeframe = "6h"
leverage = 1.0