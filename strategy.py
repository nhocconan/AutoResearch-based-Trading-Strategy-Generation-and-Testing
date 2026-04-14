#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour RSI reversal with 1-day trend filter (EMA200) and volume confirmation
# Long when RSI(14) crosses above 30 from below AND price > daily EMA200 AND volume > 1.5x 20-period average
# Short when RSI(14) crosses below 70 from above AND price < daily EMA200 AND volume > 1.5x 20-period average
# Exit when RSI crosses back to neutral (50) or opposite extreme
# This captures mean reversion in strong trends while avoiding counter-trend trades
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (14 for RSI + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        rsi_now = rsi[i]
        rsi_prev = rsi[i-1]
        
        if position == 0:
            # Long setup: RSI crosses above 30 + price > daily EMA200 + volume confirmation
            if (rsi_prev <= 30 and rsi_now > 30 and price > ema200_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: RSI crosses below 70 + price < daily EMA200 + volume confirmation
            elif (rsi_prev >= 70 and rsi_now < 70 and price < ema200_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses below 50 (mean reversion complete)
            if rsi_prev >= 50 and rsi_now < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses above 50 (mean reversion complete)
            if rsi_prev <= 50 and rsi_now > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_RSI_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0