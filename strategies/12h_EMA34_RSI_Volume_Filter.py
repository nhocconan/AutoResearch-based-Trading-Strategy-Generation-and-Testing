#!/usr/bin/env python3
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
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(34) for trend
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss.rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA34
        trend_up = close[i] > ema_34_aligned[i]
        trend_down = close[i] < ema_34_aligned[i]
        
        # Momentum filter: RSI in favorable range (not extreme)
        rsi_bullish = rsi_aligned[i] > 50 and rsi_aligned[i] < 70
        rsi_bearish = rsi_aligned[i] < 50 and rsi_aligned[i] > 30
        
        # Volume filter: above average volume
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_filter = volume[i] > vol_ma[i]
        
        # Entry conditions
        long_entry = trend_up and rsi_bullish and vol_filter
        short_entry = trend_down and rsi_bearish and vol_filter
        
        # Exit conditions: opposite conditions or volatility spike
        long_exit = not trend_up or not rsi_bullish or (atr_14_aligned[i] > 2.0 * pd.Series(atr_14_aligned).rolling(window=10, min_periods=10).mean().values[i])
        short_exit = not trend_down or not rsi_bearish or (atr_14_aligned[i] > 2.0 * pd.Series(atr_14_aligned).rolling(window=10, min_periods=10).mean().values[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_EMA34_RSI_Volume_Filter"
timeframe = "12h"
leverage = 1.0