#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation
# - 4h trend: price above/below 4h EMA50
# - 1d trend: price above/below 1d EMA200 for stronger bias
# - Entry: 1h RSI(14) crosses above 55 (long) or below 45 (short) with volume > 1.3x 20-period average
# - Exit: RSI crosses back below 45 (long) or above 55 (short) OR trend reversal
# - Session filter: 08-20 UTC only
# - Position size: 0.20
# - Target: 15-30 trades/year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA50
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d data
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) on 1h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: RSI crosses above 55 + volume surge + 4h/1d uptrend
            if (rsi[i] > 55 and rsi[i-1] <= 55 and 
                volume[i] > 1.3 * vol_ma[i] and
                close[i] > ema_50_4h_aligned[i] and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            
            # Short entry: RSI crosses below 45 + volume surge + 4h/1d downtrend
            elif (rsi[i] < 45 and rsi[i-1] >= 45 and 
                  volume[i] > 1.3 * vol_ma[i] and
                  close[i] < ema_50_4h_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses below 45 OR trend reversal
            if (rsi[i] < 45 and rsi[i-1] >= 45) or \
               close[i] < ema_50_4h_aligned[i] or \
               close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI crosses above 55 OR trend reversal
            if (rsi[i] > 55 and rsi[i-1] <= 55) or \
               close[i] > ema_50_4h_aligned[i] or \
               close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_TrendFilter_Volume_Session"
timeframe = "1h"
leverage = 1.0