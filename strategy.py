#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation
# Long when RSI < 30 (oversold) + price above 4h EMA50 (uptrend) + volume spike
# Short when RSI > 70 (overbought) + price below 4h EMA50 (downtrend) + volume spike
# Uses 4h EMA for trend direction, 1h for entry timing precision
# Target: 60-150 total trades over 4 years (15-37/year). Timeframe: 1h, HTF: 4h.
# Session filter: 08-20 UTC to avoid low-volume Asian session noise.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 50-period EMA on 4h
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Pre-calculate session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or not in_session[i]):
            continue
        
        # Volume condition: current volume > 1.5 * 20-period median
        vol_median = np.median(volume[max(0, i-19):i+1])
        vol_ok = volume[i] > 1.5 * vol_median
        
        # Long entry: RSI < 30 (oversold) + price above 4h EMA50 + volume spike
        if (rsi[i] < 30 and
            close[i] > ema_4h_aligned[i] and
            vol_ok and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: RSI > 70 (overbought) + price below 4h EMA50 + volume spike
        elif (rsi[i] > 70 and
              close[i] < ema_4h_aligned[i] and
              vol_ok and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral zone (40-60) or reverse signal
        elif position == 1 and (rsi[i] > 40 or rsi[i] < 30 and not vol_ok):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 60 or rsi[i] > 70 and not vol_ok):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_RSI14_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0