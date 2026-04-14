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
    
    # Load daily data for reference levels and filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day VWAP
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    cum_vol = np.cumsum(df_1d['volume'].values)
    cum_tpv = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_1d = np.where(cum_vol > 0, cum_tpv / cum_vol, np.nan)
    
    # Align daily levels to 12h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate RSI on daily close (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    rsi = np.full_like(close_1d, np.nan)
    
    # Wilder smoothing for RSI
    for i in range(len(close_1d)):
        if i == 0:
            avg_gain[i] = gain[i]
            avg_loss[i] = loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above daily high with RSI not overbought
            if close[i] > high_1d_aligned[i] and rsi_aligned[i] < 70:
                position = 1
                signals[i] = position_size
            # Short: price breaks below daily low with RSI not oversold
            elif close[i] < low_1d_aligned[i] and rsi_aligned[i] > 30:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below daily VWAP
            if close[i] < vwap_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above daily VWAP
            if close[i] > vwap_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_PriceBreak_VWAP_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0