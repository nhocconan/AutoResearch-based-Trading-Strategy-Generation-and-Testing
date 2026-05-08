#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h price action with 4h/1d trend filters and volume confirmation.
# Long when: price > 4h EMA50, price > 1d EMA200, RSI(14) crosses above 50, and volume > 1.5x 20-period average.
# Short when: price < 4h EMA50, price < 1d EMA200, RSI(14) crosses below 50, and volume > 1.5x 20-period average.
# Exit when RSI crosses back below 40 (long) or above 60 (short) to avoid whipsaws.
# Uses multiple timeframe alignment for trend direction and 1h for precise entry timing.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

name = "1h_Trend_RSI_Volume_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate EMA200 on 1d close
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for EMA200 and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: above both EMAs, RSI crosses above 50, volume spike
            long_cond = (close[i] > ema50_4h_aligned[i]) and (close[i] > ema200_1d_aligned[i]) and \
                        (rsi[i] > 50) and (rsi[i-1] <= 50) and volume_filter[i]
            # Short conditions: below both EMAs, RSI crosses below 50, volume spike
            short_cond = (close[i] < ema50_4h_aligned[i]) and (close[i] < ema200_1d_aligned[i]) and \
                         (rsi[i] < 50) and (rsi[i-1] >= 50) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI crosses below 40 (overbought exit)
            if rsi[i] < 40 and rsi[i-1] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI crosses above 60 (oversold exit)
            if rsi[i] > 60 and rsi[i-1] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals