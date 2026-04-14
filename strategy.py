#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(14) mean reversion with 4-hour trend filter and volume confirmation
# Long when 1h RSI < 30, price > 4h EMA200 (bullish trend), and volume > 1.5x 20-period average
# Short when 1h RSI > 70, price < 4h EMA200 (bearish trend), and volume > 1.5x 20-period average
# Exit when RSI crosses back to neutral (40 for long exit, 60 for short exit)
# Uses 4h for trend direction and 1h for precise entry timing to reduce whipsaw
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h EMA200 to 1h timeframe
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 200  # for 200-period EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long setup: RSI oversold, price above 4h EMA200, volume confirmation
            if (rsi_val < 30 and 
                price > ema_200_4h_aligned[i] and 
                vol_current > 1.5 * vol_ma[i]):
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought, price below 4h EMA200, volume confirmation
            elif (rsi_val > 70 and 
                  price < ema_200_4h_aligned[i] and 
                  vol_current > 1.5 * vol_ma[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 40 (mean reversion complete)
            if rsi_val > 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 60 (mean reversion complete)
            if rsi_val < 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI14_4hEMA200_Volume_MeanReversion"
timeframe = "1h"
leverage = 1.0