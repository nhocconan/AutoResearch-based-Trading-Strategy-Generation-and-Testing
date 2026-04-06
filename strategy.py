#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h mean reversion with 12h trend filter and volume confirmation
# Enter long when: RSI(14) < 30, price > 12h EMA(50), volume > 1.5x avg
# Enter short when: RSI(14) > 70, price < 12h EMA(50), volume > 1.5x avg
# Exit when RSI returns to neutral zone (40-60) or opposite extreme is reached
# Uses 12h trend to filter counter-trend trades in strong moves, targeting 100-180 trades over 4 years

name = "4h_rsi_meanrev_12hema_vol_v1"
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
    
    # RSI(14) on 4h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for EMA to stabilize
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: RSI > 60 OR RSI < 30 (deep oversold) OR price < 12h EMA(50)
            if rsi[i] > 60 or rsi[i] < 30 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI < 40 OR RSI > 70 (deep overbought) OR price > 12h EMA(50)
            if rsi[i] < 40 or rsi[i] > 70 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: RSI extreme + trend filter + volume
            if rsi[i] < 30 and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Oversold but above 12h EMA - bullish mean reversion
                signals[i] = 0.25
                position = 1
            elif rsi[i] > 70 and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Overbought but below 12h EMA - bearish mean reversion
                signals[i] = -0.25
                position = -1
    
    return signals