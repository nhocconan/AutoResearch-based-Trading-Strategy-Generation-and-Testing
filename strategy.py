#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w RSI filter
# Long when price breaks above 4h Donchian upper band (20-bar high) with above-average 1d volume and 1w RSI < 70
# Short when price breaks below 4h Donchian lower band (20-bar low) with above-average 1d volume and 1w RSI > 30
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in bull markets via breakout momentum and in bear markets via mean-reversion breakouts.

name = "4h_donchian20_1d_volume_1w_rsi_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_1d_avg = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1w data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_1d_avg[i]) or np.isnan(rsi_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 20-period average
        vol_confirm = volume[i] > vol_1d_avg[i]
        
        # Long: breakout above upper band with volume and RSI not overbought
        if (close[i] > highest_high[i] and vol_confirm and 
            rsi_1w_aligned[i] < 70):
            signals[i] = 0.25
        # Short: breakout below lower band with volume and RSI not oversold
        elif (close[i] < lowest_low[i] and vol_confirm and 
              rsi_1w_aligned[i] > 30):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals