#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d trend alignment and volume confirmation
# Go long when: RSI > 55, price > 4h VWAP, volume > 1.3x avg, during active session (08-20 UTC)
# Go short when: RSI < 45, price < 4h VWAP, volume > 1.3x avg, during active session
# Exit when RSI crosses opposite threshold (50 for longs, 50 for shorts)
# Uses 4h trend filter to avoid counter-trend trades, targeting 80-120 trades over 4 years

name = "1h_momentum_4hvwap_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h VWAP for trend filter
    df_4h = get_htf_data(prices, '4h')
    typical_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3
    vwap_4h = (np.cumsum(typical_4h * df_4h['volume'].values) / 
               np.cumsum(df_4h['volume'].values))
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Volume confirmation: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.3 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(vwap_4h_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: RSI < 50 (momentum faded)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI > 50 (momentum faded)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI momentum + trend filter + volume + session
            if in_session and volume[i] > volume_threshold[i]:
                if rsi[i] > 55 and close[i] > vwap_4h_aligned[i]:
                    # Bullish momentum above 4h VWAP
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] < 45 and close[i] < vwap_4h_aligned[i]:
                    # Bearish momentum below 4h VWAP
                    signals[i] = -0.20
                    position = -1
    
    return signals