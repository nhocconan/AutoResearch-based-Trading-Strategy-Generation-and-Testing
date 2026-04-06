#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and daily volume spike
# Long when RSI < 30, 4h close above 50 EMA, and daily volume > 1.5x 20-day average
# Short when RSI > 70, 4h close below 50 EMA, and daily volume > 1.5x 20-day average
# Uses 4h for trend direction, daily for volume confirmation, 1h for entry timing.
# Target: 60-150 total trades over 4 years (15-37/year) to stay within optimal range.

name = "1h_rsi14_4h_trend_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h trend filter: EMA(50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Daily volume filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_ratio = volume_1d / (vol_ma_20_aligned + 1e-10)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Skip if 4h EMA or daily volume data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: RSI reverts to mean or trend changes
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
            # Look for entries with volume confirmation
            # Long: RSI < 30, 4h uptrend, volume spike
            if (rsi[i] < 30 and 
                close[i] > ema_4h_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70, 4h downtrend, volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema_4h_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.20
                position = -1
    
    return signals