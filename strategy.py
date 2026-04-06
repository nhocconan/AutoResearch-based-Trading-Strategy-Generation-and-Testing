#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion strategy using 4h trend filter and 1d volatility filter
# Long when price pulls back to 4h EMA20 during uptrend + 1d volatility low
# Short when price rallies to 4h EMA20 during downtrend + 1d volatility low
# Uses RSI(14) for entry timing with oversold/overbought levels
# Targets 60-150 trades over 4 years by combining trend alignment with mean reversion
# Works in both bull and bear markets by following 4h trend direction

name = "1h_meanrev_4htrend_1dvol_v1"
timeframe = "1h"
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
    
    # 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d ATR percentage (normalized volatility)
    atr_pct_1d = atr_1d_aligned / close
    vol_threshold = 0.02  # 2% daily volatility threshold
    
    # 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr_pct_1d[i]) or 
            np.isnan(rsi[i]) or np.isnan(hours[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: RSI returns to neutral or trend reversal
        if position == 1:  # long position
            if rsi[i] >= 50 or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if rsi[i] <= 50 or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries with trend and volatility filters
            # Bullish setup: price near 4h EMA20 uptrend + oversold RSI + low volatility
            if (close[i] <= ema_4h_aligned[i] * 1.005 and  # within 0.5% of EMA
                rsi[i] < 30 and 
                ema_4h_aligned[i] > ema_4h_aligned[i-1] and  # 4h EMA rising
                atr_pct_1d[i] < vol_threshold):
                signals[i] = 0.20
                position = 1
            # Bearish setup: price near 4h EMA20 downtrend + overbought RSI + low volatility
            elif (close[i] >= ema_4h_aligned[i] * 0.995 and  # within 0.5% of EMA
                  rsi[i] > 70 and 
                  ema_4h_aligned[i] < ema_4h_aligned[i-1] and  # 4h EMA falling
                  atr_pct_1d[i] < vol_threshold):
                signals[i] = -0.20
                position = -1
    
    return signals