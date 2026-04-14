#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index + 1-week RSI mean reversion with volume filter
# Long when weekly RSI < 30 AND choppiness > 61.8 (ranging) AND volume > 1.5x average
# Short when weekly RSI > 70 AND choppiness > 61.8 (ranging) AND volume > 1.5x average
# Exit when RSI crosses back to neutral zone (40-60)
# This captures mean reversion in ranging markets while avoiding trending periods
# Target: 75-200 total trades over 4 years (19-50/year) with strict entry conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for RSI
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI on weekly close (14-period)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate Choppiness Index on 4h (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr_14 * 14) / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        rsi = rsi_1w_aligned[i]
        
        if position == 0:
            # Long setup: RSI oversold + ranging market + volume confirmation
            if (rsi < 30 and chop[i] > 61.8 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought + ranging market + volume confirmation
            elif (rsi > 70 and chop[i] > 61.8 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or overbought
            if rsi >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral or oversold
            if rsi <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_WeeklyRSI_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0