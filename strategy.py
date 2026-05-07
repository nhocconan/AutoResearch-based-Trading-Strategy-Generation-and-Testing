# 1D_WeeklyMATrend_RSIFilter
# Hypothesis: Daily RSI oversold/overbought with weekly trend filter. Uses weekly MA to avoid counter-trend trades in both bull and bear markets. RSI extremes provide mean-reversion entries aligned with weekly trend. Targets 10-25 trades/year to minimize fee drag. Uses discrete position sizing (0.25).
name = "1D_WeeklyMATrend_RSIFilter"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
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
    
    start_idx = max(14, 34)  # Ensure we have RSI and weekly MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + weekly uptrend (price > weekly EMA34)
            if (rsi[i] < 30 and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + weekly downtrend (price < weekly EMA34)
            elif (rsi[i] > 70 and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60)
            if (40 <= rsi[i] <= 60):
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals