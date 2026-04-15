#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and volume confirmation
# Uses extreme RSI(2) readings for mean reversion entries, filtered by 4h EMA50 trend
# and confirmed by volume spikes. Works in bull/bear by only taking reversals
# in the direction of the 4h trend. Target: 60-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate RSI(2) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA50 on 4h for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period on 1h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_avg[i])):
            continue
        
        # Long entry: RSI(2) < 10 (oversold) + volume spike + price above 4h EMA50
        if (rsi[i] < 10 and
            volume[i] > 1.5 * vol_avg[i] and
            close[i] > ema50_4h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: RSI(2) > 90 (overbought) + volume spike + price below 4h EMA50
        elif (rsi[i] > 90 and
              volume[i] > 1.5 * vol_avg[i] and
              close[i] < ema50_4h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral (40-60) or reverse signal
        elif position == 1 and (rsi[i] > 40 or rsi[i] < 60):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 60 or rsi[i] > 40):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSI2_MeanReversion_4hTrend"
timeframe = "1h"
leverage = 1.0