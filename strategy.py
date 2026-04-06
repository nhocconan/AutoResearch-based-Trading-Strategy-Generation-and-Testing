#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
# Long when RSI < 30 and price > 4h EMA50 with volume > 1.5x average.
# Short when RSI > 70 and price < 4h EMA50 with volume > 1.5x average.
# Uses 4h EMA50 for trend direction to avoid counter-trend trades.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "1h_rsi30_70_4h_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = gain_ma / loss_ma.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if 4h EMA data not available
        if np.isnan(ema_4h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > volume_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI > 50 or price crosses below 4h EMA50
            if (rsi[i] > 50 or 
                close[i] < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 50 or price crosses above 4h EMA50
            if (rsi[i] < 50 or 
                close[i] > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and RSI extremes
            if volume_filter:
                # Long: RSI < 30 and price > 4h EMA50
                if (rsi[i] < 30 and 
                    close[i] > ema_4h_aligned[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: RSI > 70 and price < 4h EMA50
                elif (rsi[i] > 70 and 
                      close[i] < ema_4h_aligned[i]):
                    signals[i] = -0.20
                    position = -1
    
    return signals