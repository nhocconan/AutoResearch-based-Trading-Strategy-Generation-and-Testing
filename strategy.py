#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h/1d trend filter and volume confirmation.
# Long when RSI < 30 AND price > 4h EMA50 AND price > 1d EMA200 AND volume > 1.5x 20-period avg.
# Short when RSI > 70 AND price < 4h EMA50 AND price < 1d EMA200 AND volume > 1.5x 20-period avg.
# Exit when RSI crosses back to neutral (40 for long, 60 for short).
# Uses 4h/1d for trend direction, 1h only for entry timing via RSI.
# Session filter: 08-20 UTC to reduce noise.
# Target: 100-150 total trades over 4 years (25-38/year) to balance opportunity and fee drag.

name = "1h_RSI_MeanReversion_4h1dTrend_Volume"
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
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # RSI(14) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI oversold, price above both EMAs, volume filter
            long_cond = (rsi[i] < 30) and (close[i] > ema_50_4h_aligned[i]) and \
                        (close[i] > ema_200_1d_aligned[i]) and volume_filter[i]
            # Short conditions: RSI overbought, price below both EMAs, volume filter
            short_cond = (rsi[i] > 70) and (close[i] < ema_50_4h_aligned[i]) and \
                         (close[i] < ema_200_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back to 40 (mean reversion complete)
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI crosses back to 60 (mean reversion complete)
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals