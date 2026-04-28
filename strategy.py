#!/usr/bin/env python3
"""
1h_Momentum_RSI_Trend_4hFilter
Hypothesis: 1h momentum strategy using RSI(14) with 4h EMA50 trend filter and volume confirmation.
Focuses on high-probability momentum pulls back in trending markets. Uses 4h trend for direction,
1h RSI for entry timing, and volume to filter false signals. Designed for 15-30 trades/year
to minimize fee drag while capturing trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    h4_uptrend = close > ema_50_4h_aligned
    h4_downtrend = close < ema_50_4h_aligned
    
    # 1h RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: RSI momentum with trend alignment and volume
        # Long: RSI < 40 (pullback in uptrend) + volume surge
        long_entry = (rsi[i] < 40 and 
                     h4_uptrend[i] and 
                     volume_surge[i])
        
        # Short: RSI > 60 (pullback in downtrend) + volume surge
        short_entry = (rsi[i] > 60 and 
                      h4_downtrend[i] and 
                      volume_surge[i])
        
        # Exit when RSI returns to neutral zone
        long_exit = rsi[i] > 60
        short_exit = rsi[i] < 40
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.20  # Reverse to short if criteria met, else flat
            if h4_downtrend[i] and rsi[i] > 60 and volume_surge[i]:
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif short_exit and position == -1:
            signals[i] = 0.20   # Reverse to long if criteria met, else flat
            if h4_uptrend[i] and rsi[i] < 40 and volume_surge[i]:
                position = 1
            else:
                signals[i] = 0.0
                position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Momentum_RSI_Trend_4hFilter"
timeframe = "1h"
leverage = 1.0