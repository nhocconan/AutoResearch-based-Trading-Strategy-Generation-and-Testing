# 1d_RSI_Trend_With_WeeklyFilter
# Hypothesis: Long-term trend following with RSI mean-reversion entries on daily timeframe.
# Uses weekly EMA50 for trend filter and daily RSI(14) for mean-reversion entries.
# Designed for low trade frequency (10-30/year) to minimize fee drag.
# Works in bull markets by following trend and in bear markets by fading overextended moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for weekly EMA50 and RSI
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: RSI oversold (<30) in uptrend
            if rsi[i] < 30 and trend_up:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) in downtrend
            elif rsi[i] > 70 and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or trend reversal
            if rsi[i] > 70 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or trend reversal
            if rsi[i] < 30 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_Trend_With_WeeklyFilter"
timeframe = "1d"
leverage = 1.0