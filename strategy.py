# Solution
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h_4h1d_trend_volume_v1
# Hypothesis: Use 4h and 1d timeframes for trend direction, 1h for entry timing with volume confirmation.
# In both bull and bear markets: trade with the higher timeframe trend when price pulls back to
# the 4h EMA on the 1h chart with above-average volume. This reduces whipsaws and focuses on
# higher probability entries. Target: 15-37 trades/year (60-150 over 4 years).

name = "1h_4h1d_trend_volume_v1"
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
    
    # Get 4h and 1d data for trend
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA for trend direction (20-period)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA for longer-term trend (50-period)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 1.5x 20-period average on 1h
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend: both 4h and 1d EMA must agree
        bullish = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        bearish = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        # Entry conditions with volume confirmation
        if bullish and vol_filter[i]:
            signals[i] = 0.20  # Long 20%
        elif bearish and vol_filter[i]:
            signals[i] = -0.20  # Short 20%
        else:
            signals[i] = 0.0  # Flat
    
    return signals