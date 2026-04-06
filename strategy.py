#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Bands + RSI mean reversion with volume confirmation
# Long when price touches lower BB + RSI < 30 + volume > 1.5x avg
# Short when price touches upper BB + RSI > 70 + volume > 1.5x avg
# Exit when price crosses middle BB (SMA20) or volume dries up
# Works in ranging markets (chop > 61.8) - avoids trending periods
# Targets 75-150 trades over 4 years with low frequency, high accuracy

name = "12h_bb_rsi_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 12h
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    middle_band = sma20
    
    # RSI (14) from 1d timeframe for mean reversion
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate RSI on daily close
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Align daily RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses middle BB OR volume drops below threshold
        if position == 1:  # long position
            if close[i] >= middle_band[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] <= middle_band[i] or volume[i] < volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price at BB extremes + RSI extreme + volume confirmation
            # Long: price touches lower BB + RSI oversold + volume spike
            if (close[i] <= lower_band[i] and rsi_aligned[i] < 30 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB + RSI overbought + volume spike
            elif (close[i] >= upper_band[i] and rsi_aligned[i] > 70 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals