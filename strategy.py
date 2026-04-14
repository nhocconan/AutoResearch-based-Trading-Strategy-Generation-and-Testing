#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 12h VWAP (typical price * volume / cumulative volume)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    cum_vol_12h = np.cumsum(volume_12h)
    cum_tpv_12h = np.cumsum(typical_price_12h * volume_12h)
    vwap_12h = np.where(cum_vol_12h > 0, cum_tpv_12h / cum_vol_12h, np.nan)
    
    # Calculate weekly EMA20 for trend filter
    if len(close_1w) < 20:
        return np.zeros(n)
    
    ema20_1w = np.full_like(close_1w, np.nan)
    alpha = 2.0 / (20 + 1)
    for i in range(len(close_1w)):
        if np.isnan(close_1w[i]):
            ema20_1w[i] = np.nan
        elif i == 0:
            ema20_1w[i] = close_1w[i]
        else:
            ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    
    # Align 12h levels and weekly EMA to 12h timeframe
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate 12h ATR for volatility filter
    if len(high_12h) < 14:
        return np.zeros(n)
    
    tr_12h = np.zeros_like(high_12h)
    for i in range(1, len(high_12h)):
        if np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or np.isnan(high_12h[i-1]) or np.isnan(low_12h[i-1]):
            tr_12h[i] = np.nan
        else:
            tr_12h[i] = max(high_12h[i] - low_12h[i], 
                           abs(high_12h[i] - high_12h[i-1]), 
                           abs(low_12h[i] - low_12h[i-1]))
    
    atr_12h = np.full_like(high_12h, np.nan)
    if len(high_12h) >= 14:
        atr_12h[13] = np.nanmean(tr_12h[1:14])
        for i in range(14, len(high_12h)):
            if np.isnan(tr_12h[i]):
                atr_12h[i] = atr_12h[i-1]
            else:
                atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size to limit trades
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_12h_aligned[i]) or np.isnan(low_12h_aligned[i]) or
            np.isnan(vwap_12h_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        vol_ma_20 = np.full_like(volume_12h, np.nan)
        for j in range(19, len(volume_12h)):
            vol_ma_20[j] = np.mean(volume_12h[j-19:j+1])
        
        vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
        if np.isnan(vol_ma_20_aligned[i]) or vol_ma_20_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume_12h_aligned[i] / vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for long entries: breakout above 12h high with volume surge in bullish weekly trend
            if (close[i] > high_12h_aligned[i] and 
                volume_ratio > 2.0 and  # Volume surge
                close[i] > ema20_1w_aligned[i]):  # Above weekly EMA20 (bullish trend)
                position = 1
                signals[i] = position_size
            # Look for short entries: breakdown below 12h low with volume surge in bearish weekly trend
            elif (close[i] < low_12h_aligned[i] and 
                  volume_ratio > 2.0 and
                  close[i] < ema20_1w_aligned[i]):  # Below weekly EMA20 (bearish trend)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 12h VWAP or volatility drops
            if (close[i] < vwap_12h_aligned[i] or
                volume_ratio < 1.0):  # Low volume indicates weakening trend
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 12h VWAP or volatility drops
            if (close[i] > vwap_12h_aligned[i] or
                volume_ratio < 1.0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_VWAP_Breakout_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0