#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation
# Long when: price > 4h EMA(50), RSI(14) > 50, volume > 1.5x 20-period average
# Short when: price < 4h EMA(50), RSI(14) < 50, volume > 1.5x 20-period average
# Uses 4h EMA for trend direction, 1h for entry timing with volume confirmation
# Target: 60-150 total trades over 4 years (15-37/year) with controlled risk

name = "1h_momentum_4h_ema_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h indicators
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price below 4h EMA or RSI < 40
            if close[i] < ema_4h_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price above 4h EMA or RSI > 60
            if close[i] > ema_4h_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                # Long when price above 4h EMA and RSI > 50
                if close[i] > ema_4h_aligned[i] and rsi[i] > 50:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short when price below 4h EMA and RSI < 50
                elif close[i] < ema_4h_aligned[i] and rsi[i] < 50:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals