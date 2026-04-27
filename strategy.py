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
    
    # Get weekly data for indicators (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1w = 100 - (100 / (1 + rs))
    
    # Calculate weekly ATR(14) for volatility
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr = np.maximum(high_1w - low_1w, 
                    np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                               np.abs(low_1w - np.roll(close_1w, 1))))
    tr[0] = high_1w[0] - low_1w[0]
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly indicators to daily
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Get daily price for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute day of week filter (Mon-Thu)
    days = pd.DatetimeIndex(prices['open_time']).dayofweek  # Mon=0, Thu=3
    
    # Warmup: need all indicators
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_14_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Day filter: only trade Monday-Thursday
        day = days[i]
        if day > 3:  # Fri=4, Sat=5, Sun=6
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_14_1w_aligned[i]
        atr_val = atr_14_1w_aligned[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        
        # Volatility filter: ATR > 20-period median (high volatility regime)
        if i >= 20:
            atr_ma = pd.Series(atr_14_1w_aligned[:i+1]).rolling(window=20, min_periods=20).median().iloc[-1]
        else:
            atr_ma = atr_val
        vol_filter = atr_val > atr_ma
        
        # Entry conditions
        if position == 0:
            # Long: weekly RSI < 30 (oversold) + price breaks above Donchian high + volatility
            if rsi_val < 30 and close[i] > upper_band and vol_filter:
                signals[i] = size
                position = 1
            # Short: weekly RSI > 70 (overbought) + price breaks below Donchian low + volatility
            elif rsi_val > 70 and close[i] < lower_band and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 or price hits lower band
            if rsi_val > 50 or close[i] < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 50 or price hits upper band
            if rsi_val < 50 or close[i] > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyRSI_DonchianBreakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0