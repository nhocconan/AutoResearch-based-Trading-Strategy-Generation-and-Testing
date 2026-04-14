#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 3-period RSI with 1-day Bollinger Bands mean reversion
# Long when RSI(3) < 15 AND price < lower Bollinger Band(20,2) on 1d AND volume > 1.5x 20-period average
# Short when RSI(3) > 85 AND price > upper Bollinger Band(20,2) on 1d AND volume > 1.5x 20-period average
# Exit when RSI(3) crosses back above 50 (long) or below 50 (short)
# Uses extreme RSI on lower timeframe for timing, Bollinger Bands on higher timeframe for overextension
# Target: 100-200 total trades over 4 years (25-50/year) with mean reversion edge in both bull/bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI(3) on price closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/3, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/3, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands on 1d (20,2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: RSI oversold AND price below lower BB AND volume confirmation
            if (rsi_val < 15 and price < lower_bb_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought AND price above upper BB AND volume confirmation
            elif (rsi_val > 85 and price > upper_bb_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses back above 50
            if rsi_val > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses back below 50
            if rsi_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_RSI3_BBands20_2_Volume"
timeframe = "4h"
leverage = 1.0