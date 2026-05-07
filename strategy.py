#!/usr/bin/env python3
name = "4h_MultiFactor_Confluence_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for higher timeframe trend and volatility filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA20 for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate 12h ATR15 for volatility filter
    tr_12h = np.maximum(high_12h - low_12h, 
                        np.maximum(abs(high_12h - np.roll(close_12h, 1)),
                                   abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_15_12h = pd.Series(tr_12h).rolling(window=15, min_periods=15).mean().values
    atr_15_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_15_12h)
    
    # Calculate 12h volume moving average for volume filter
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 4h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 4h Bollinger Bands(20,2) for volatility context
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(atr_15_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. 12h trend up: price above 12h EMA20
            # 2. Low volatility environment: 12h ATR below its 20-period average
            # 3. RSI not overbought: below 70
            # 4. Price near lower Bollinger Band: potential bounce
            if (close[i] > ema_20_12h_aligned[i] and
                atr_15_12h_aligned[i] < np.nanmean(atr_15_12h_aligned[max(0, i-20):i]) and
                rsi[i] < 70 and
                close[i] <= bb_lower[i] * 1.02):  # Within 2% of lower BB
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. 12h trend down: price below 12h EMA20
            # 2. Low volatility environment
            # 3. RSI not oversold: above 30
            # 4. Price near upper Bollinger Band: potential rejection
            elif (close[i] < ema_20_12h_aligned[i] and
                  atr_15_12h_aligned[i] < np.nanmean(atr_15_12h_aligned[max(0, i-20):i]) and
                  rsi[i] > 30 and
                  close[i] >= bb_upper[i] * 0.98):  # Within 2% of upper BB
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend reversal or overextension
            if (close[i] < ema_20_12h_aligned[i] or  # Trend change
                rsi[i] > 75 or                      # Overbought
                close[i] >= bb_upper[i]):           # Reached upper BB
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend reversal or overextension
            if (close[i] > ema_20_12h_aligned[i] or  # Trend change
                rsi[i] < 25 or                      # Oversold
                close[i] <= bb_lower[i]):           # Reached lower BB
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals