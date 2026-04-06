#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d RSI mean reversion + volume confirmation
# Long when CHOP > 61.8 (range) AND RSI(14) < 30 AND volume > 1.5x average
# Short when CHOP > 61.8 (range) AND RSI(14) > 70 AND volume > 1.5x average
# Exit when RSI crosses back to neutral (40-60 range) or CHOP < 38.2 (trending)
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in both bull/bear markets by fading extremes in ranging conditions

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
    
    # Choppiness Index (14-period) - measures ranging vs trending
    # High values (>61.8) indicate ranging/choppy market (good for mean reversion)
    # Low values (<38.2) indicate trending market
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    
    atr = np.array(atr_list)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum()
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.values
    
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
        if np.isnan(chop[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: RSI returns to neutral range OR market starts trending
        if position == 1:  # long position
            if rsi_aligned[i] >= 40 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if rsi_aligned[i] <= 60 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging market (CHOP > 61.8) with RSI extremes
            # Long: RSI oversold (<30) in ranging market + volume confirmation
            if (chop[i] > 61.8 and rsi_aligned[i] < 30 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) in ranging market + volume confirmation
            elif (chop[i] > 61.8 and rsi_aligned[i] > 70 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals