# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d Bollinger Bands Width Percentile + RSI Mean Reversion + Volume Confirmation
Hypothesis: In ranging markets (identified by low Bollinger Bands Width percentile),
price tends to revert to the mean. RSI identifies overbought/oversold conditions,
and volume confirmation filters for institutional interest. Works in both bull and bear
markets by adapting to volatility regime. Low trade frequency due to strict percentile
threshold and volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def bollinger_bands_width_percentile(close, bb_length=20, bb_std=2.0, percentile_lookback=100):
    """Calculate Bollinger Bands Width and its percentile rank"""
    # Calculate Bollinger Bands
    sma = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    std = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    bb_width = upper - lower
    
    # Calculate percentile rank of current BB width
    bb_width_percentile = np.zeros_like(close)
    for i in range(len(close)):
        if i < percentile_lookback:
            bb_width_percentile[i] = 50.0  # Neutral when insufficient history
        else:
            # Calculate percentile of current BB width vs lookback period
            historical_widths = bb_width[max(0, i-percentile_lookback):i+1]
            current_width = bb_width[i]
            if len(historical_widths) > 1:
                # Percentile: percentage of values less than current
                bb_width_percentile[i] = (np.sum(historical_widths < current_width) / len(historical_widths)) * 100
            else:
                bb_width_percentile[i] = 50.0
    
    return bb_width_percentile, upper, lower, sma

def calculate_rsi(close, length=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    for i in range(len(close)):
        if i < length:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i) + gain[i]) / (i + 1)
                avg_loss[i] = (avg_loss[i-1] * (i) + loss[i]) / (i + 1)
        else:
            avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss[i]) / length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (optional - we'll use price vs weekly SMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA for trend filter
    close_1w = df_1w['close'].values
    sma_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Calculate Bollinger Bands Width Percentile on 1d
    bb_width_percentile, bb_upper, bb_middle, bb_lower = bollinger_bands_width_percentile(
        close, bb_length=20, bb_std=2.0, percentile_lookback=100
    )
    
    # Calculate RSI
    rsi = calculate_rsi(close, length=14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_confirmed = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(bb_width_percentile[i]) or np.isnan(rsi[i]) or np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        bb_percentile = bb_width_percentile[i]
        rsi_val = rsi[i]
        vol_ok = vol_confirmed[i]
        price = close[i]
        weekly_sma = sma_1w_aligned[i]
        
        if position == 0:
            # Enter long: low volatility (rangy market) + RSI oversold + volume confirmation
            # In ranging markets (low BB width percentile), look for mean reversion
            if (bb_percentile < 30 and  # Low volatility regime
                rsi_val < 30 and        # Oversold
                vol_ok and              # Volume confirmation
                price > weekly_sma):    # Above weekly trend (bullish bias)
                signals[i] = 0.25
                position = 1
            # Enter short: low volatility + RSI overbought + volume confirmation
            elif (bb_percentile < 30 and   # Low volatility regime
                  rsi_val > 70 and         # Overbought
                  vol_ok and               # Volume confirmation
                  price < weekly_sma):     # Below weekly trend (bearish bias)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or volatility increases (breakout)
            if rsi_val > 70 or bb_percentile > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or volatility increases (breakdown)
            if rsi_val < 30 or bb_percentile > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_BBWidth_Percentile_RSI_MeanRev_Volume"
timeframe = "1d"
leverage = 1.0