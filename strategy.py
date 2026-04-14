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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donch_high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    donch_low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Calculate weekly RSI(14) for momentum filter
    delta_w = pd.Series(df_1w['close']).diff()
    gain_w = delta_w.clip(lower=0)
    loss_w = -delta_w.clip(upper=0)
    avg_gain_w = gain_w.rolling(window=14, min_periods=14).mean()
    avg_loss_w = loss_w.rolling(window=14, min_periods=14).mean()
    rs_w = avg_gain_w / avg_loss_w
    rsi_14w = (100 - (100 / (1 + rs_w))).values
    rsi_14w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14w)
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_20w_aligned[i]) or 
            np.isnan(donch_low_20w_aligned[i]) or
            np.isnan(rsi_14w_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr_1d_aligned[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005  # Minimum 0.5% ATR relative to price
        
        if position == 0:
            # Long setup: break above weekly Donchian high + RSI > 50 + volatility filter
            if (price > donch_high_20w_aligned[i] and 
                rsi_14w_aligned[i] > 50 and 
                vol_filter):
                position = 1
                signals[i] = position_size
            # Short setup: break below weekly Donchian low + RSI < 50 + volatility filter
            elif (price < donch_low_20w_aligned[i] and 
                  rsi_14w_aligned[i] < 50 and 
                  vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly Donchian low
            if price < donch_low_20w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly Donchian high
            if price > donch_high_20w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wDonchian20_RSI50_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0