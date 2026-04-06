#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI + 4h MACD + volume confirmation with session filter
# Enter long when: RSI(14) > 60, MACD(12,26,9) bullish crossover on 4h, volume > 1.5x average, during active session (08-20 UTC)
# Enter short when: RSI(14) < 40, MACD bearish crossover on 4h, volume > 1.5x average, during active session
# Uses momentum confirmation from higher timeframe to filter false signals on 1h
# Exit when RSI returns to neutral zone (40-60) or opposite MACD crossover occurs
# Target: 60-150 trades over 4 years by combining multiple filters

name = "1h_rsi_macd_vol_session_v1"
timeframe = "1h"
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
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # MACD on 4h
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_12 = pd.Series(close_4h).ewm(span=12, adjust=False).mean().values
    ema_26 = pd.Series(close_4h).ewm(span=26, adjust=False).mean().values
    macd = ema_12 - ema_26
    signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    macd_hist = macd - signal_line
    macd_hist_aligned = align_htf_to_ltf(prices, df_4h, macd_hist)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Wait for MACD to stabilize
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(macd_hist_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: RSI < 40 OR MACD histogram turns negative
            if rsi[i] < 40 or macd_hist_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI > 60 OR MACD histogram turns positive
            if rsi[i] > 60 or macd_hist_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme + MACD confirmation + volume + session
            if in_session and volume[i] > volume_threshold[i]:
                if rsi[i] > 60 and macd_hist_aligned[i] > 0 and macd_hist_aligned[i-1] <= 0:
                    # RSI overbought with fresh bullish MACD crossover
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] < 40 and macd_hist_aligned[i] < 0 and macd_hist_aligned[i-1] >= 0:
                    # RSI oversold with fresh bearish MACD crossover
                    signals[i] = -0.20
                    position = -1
    
    return signals