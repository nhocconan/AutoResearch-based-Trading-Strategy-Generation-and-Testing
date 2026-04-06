# This strategy implements a 12h timeframe approach using:
# - 1-week Choppiness Index to detect ranging markets (high values = range)
# - 1-day RSI for mean reversion signals (overbought/oversold)
# - Volume confirmation to filter false signals
# The idea is to fade extremes in ranging markets (Choppiness > 61.8) with volume confirmation
# Works in both bull and bear markets by focusing on mean reversion during consolidation periods
# Target: 50-150 trades over 4 years (12-37/year)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_chop_rsi_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period) from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate True Range for weekly data
    tr_list = []
    for i in range(len(weekly_close)):
        if i == 0:
            tr = weekly_high[i] - weekly_low[i]
        else:
            tr = max(weekly_high[i] - weekly_low[i], 
                    abs(weekly_high[i] - weekly_close[i-1]), 
                    abs(weekly_low[i] - weekly_close[i-1]))
        tr_list.append(tr)
    
    tr = np.array(tr_list)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    highest_high = pd.Series(weekly_high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(weekly_low).rolling(window=14, min_periods=14).min()
    
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.values
    
    # Align weekly Choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # RSI (14-period) from 1d timeframe for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate RSI on daily close
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Align daily RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(chop_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: RSI returns to neutral range OR market starts trending
        if position == 1:  # long position
            if rsi_aligned[i] >= 40 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if rsi_aligned[i] <= 60 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging market (CHOP > 61.8) with RSI extremes
            # Long: RSI oversold (<30) in ranging market + volume confirmation
            if (chop_aligned[i] > 61.8 and rsi_aligned[i] < 30 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) in ranging market + volume confirmation
            elif (chop_aligned[i] > 61.8 and rsi_aligned[i] > 70 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals