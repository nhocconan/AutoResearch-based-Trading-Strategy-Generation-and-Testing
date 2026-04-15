#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume spike
# Uses 4h EMA for trend direction, 1h RSI for momentum, and volume spike for confirmation.
# Only trades during active session (08-20 UTC). Works in bull/bear via trend following.
# Target: 60-150 total trades over 4 years = 15-37/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 2.0 * median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    vol_spike = volume > (2.0 * vol_median)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(20, n):
        if not in_session[i]:
            continue
        
        # Skip if EMA not ready
        if np.isnan(ema_4h_aligned[i]):
            continue
        
        # Long: uptrend (price > EMA) + RSI > 50 + volume spike
        if (close[i] > ema_4h_aligned[i] and
            rsi[i] > 50 and
            vol_spike[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: downtrend (price < EMA) + RSI < 50 + volume spike
        elif (close[i] < ema_4h_aligned[i] and
              rsi[i] < 50 and
              vol_spike[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend reversal or RSI extreme
        elif position == 1 and (close[i] < ema_4h_aligned[i] or rsi[i] < 30):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_4h_aligned[i] or rsi[i] > 70):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_EMA_RSI_Volume_Spike"
timeframe = "1h"
leverage = 1.0