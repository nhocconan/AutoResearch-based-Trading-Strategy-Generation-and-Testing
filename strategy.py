# Hypothesis: 1d strategy using weekly Bollinger Bands breakout with volume confirmation and ADX trend filter.
# Weekly Bollinger Bands provide dynamic support/resistance; breakouts with volume and trend strength
# capture sustained moves in both bull and bear markets. Weekly timeframe reduces noise and false signals.
# Position size 0.25 to manage drawdown. Target trades: 20-50 over 4 years to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Bollinger Bands (20-period, 2 std)
    bb_length = 20
    bb_mult = 2.0
    basis_1w = np.full(len(df_1w), np.nan)
    upper_1w = np.full(len(df_1w), np.nan)
    lower_1w = np.full(len(df_1w), np.nan)
    
    if len(df_1w) >= bb_length:
        # Calculate SMA and standard deviation
        sma = np.full(len(df_1w), np.nan)
        for i in range(bb_length - 1, len(df_1w)):
            sma[i] = np.mean(close_1w[i - bb_length + 1:i + 1])
        
        # Calculate standard deviation
        std_dev = np.full(len(df_1w), np.nan)
        for i in range(bb_length - 1, len(df_1w)):
            dev = close_1w[i - bb_length + 1:i + 1] - sma[i]
            std_dev[i] = np.sqrt(np.mean(dev * dev))
        
        basis_1w = sma
        upper_1w = basis_1w + bb_mult * std_dev
        lower_1w = basis_1w - bb_mult * std_dev
    
    # Calculate weekly ADX (14-period)
    adx_length = 14
    high_low = high_1w[1:] - high_1w[:-1]
    low_high = low_1w[:-1] - low_1w[1:]
    
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period - 1] = np.mean(data[1:period])
            for i in range(period, len(data)):
                result[i] = (result[i - 1] * (period - 1) + data[i]) / period
        return result
    
    if len(df_1w) >= adx_length:
        atr_1w = wilders_smoothing(tr, adx_length)
        plus_di_1w = 100 * wilders_smoothing(plus_dm, adx_length) / atr_1w
        minus_di_1w = 100 * wilders_smoothing(minus_dm, adx_length) / atr_1w
        dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
        adx_1w = wilders_smoothing(dx_1w, adx_length)
    else:
        adx_1w = np.full(len(df_1w), np.nan)
    
    # Align weekly indicators to daily
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Daily volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_1w_aligned[i]) or
            np.isnan(lower_1w_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility (weekly BB width < 1% of price)
        bb_width = (upper_1w_aligned[i] - lower_1w_aligned[i]) / close[i]
        if bb_width < 0.01:
            signals[i] = 0.0
            continue
        
        # Skip low volume (volume < 70% of 20-day MA)
        if volume[i] < 0.7 * vol_ma[i]:
            signals[i] = 0.0
            continue
        
        # Skip weak trend (ADX < 22)
        if adx_1w_aligned[i] < 22:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above upper weekly BB
            if close[i] > upper_1w_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: Close below lower weekly BB
            elif close[i] < lower_1w_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Close below lower weekly BB OR ADX < 18
            if close[i] < lower_1w_aligned[i] or adx_1w_aligned[i] < 18:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Close above upper weekly BB OR ADX < 18
            if close[i] > upper_1w_aligned[i] or adx_1w_aligned[i] < 18:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Bollinger_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0