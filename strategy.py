#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12-hour Williams %R overbought/oversold levels and 4h RSI divergence.
# In bear markets, short when Williams %R > -20 (overbought) on 12h AND RSI divergence on 4h.
# In bull markets, long when Williams %R < -80 (oversold) on 12h AND RSI divergence on 4h.
# Volume > 1.5x 20-period average confirms momentum.
# Uses Williams %R for mean reversion in extremes and RSI divergence for momentum confirmation.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Williams %R
    df_12h = get_htf_data(prices, '12h')
    
    # 12h Williams %R (14)
    williams_len = 14
    if len(df_12h) < williams_len:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=williams_len, min_periods=williams_len).max().values
    lowest_low = pd.Series(low_12h).rolling(window=williams_len, min_periods=williams_len).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # 4h RSI (14) for divergence
    df_4h = get_htf_data(prices, '4h')
    rsi_len = 14
    if len(df_4h) < rsi_len:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=rsi_len, adjust=False, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_len, adjust=False, min_periods=rsi_len).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    
    # RSI divergence: bullish when price makes lower low but RSI makes higher low
    # bearish when price makes higher high but RSI makes lower high
    lookback = 5
    rsi_div_bull = np.zeros(n, dtype=bool)
    rsi_div_bear = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        if not np.isnan(rsi_aligned[i]) and not np.isnan(close[i]):
            # Check for bullish divergence: price lower low, RSI higher low
            if (close[i] < close[i-lookback] and 
                rsi_aligned[i] > rsi_aligned[i-lookback]):
                rsi_div_bull[i] = True
            # Check for bearish divergence: price higher high, RSI lower high
            elif (close[i] > close[i-lookback] and 
                  rsi_aligned[i] < rsi_aligned[i-lookback]):
                rsi_div_bear[i] = True
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(williams_len*2, rsi_len*2, 20, lookback)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if volume_confirmed:
                # Long setup: Williams %R oversold (< -80) AND bullish RSI divergence
                if (williams_r_aligned[i] < -80 and 
                    rsi_div_bull[i]):
                    position = 1
                    signals[i] = position_size
                # Short setup: Williams %R overbought (> -20) AND bearish RSI divergence
                elif (williams_r_aligned[i] > -20 and 
                      rsi_div_bear[i]):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or bearish divergence
            if (williams_r_aligned[i] > -50 or 
                rsi_div_bear[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or bullish divergence
            if (williams_r_aligned[i] < -50 or 
                rsi_div_bull[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_WilliamsRSI_Divergence_Volume_v1"
timeframe = "4h"
leverage = 1.0