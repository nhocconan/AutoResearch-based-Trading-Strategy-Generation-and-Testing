# 1d_WeeklyBias_Volume_Spike
# Hypothesis: Use weekly EMA slope as trend filter (bull/bear) and daily RSI for mean reversion entries.
# In bull markets (weekly EMA up), go long when daily RSI < 30 (oversold).
# In bear markets (weekly EMA down), go short when daily RSI > 70 (overbought).
# Volume spike (>2x 20-day median) confirms conviction. Weekly EMA slope avoids whipsaws.
# Designed for low trade frequency (~10-20/year) to minimize fee drag.
# Works in bull (follow weekly trend) and bear (fade daily extremes with weekly bias).

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
    
    # Weekly EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_slope = np.diff(ema_1w, prepend=ema_1w[0])
    ema_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_slope)
    
    # Daily RSI(14) for mean reversion
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Volume spike: > 2x 20-day median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Warmup for weekly EMA
        if (np.isnan(ema_1w_slope_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: weekly uptrend, daily oversold, volume spike
        if (ema_1w_slope_aligned[i] > 0 and 
            rsi_1d_aligned[i] < 30 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: weekly downtrend, daily overbought, volume spike
        elif (ema_1w_slope_aligned[i] < 0 and 
              rsi_1d_aligned[i] > 70 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: weekly trend change or RSI returns to neutral (30-70)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (ema_1w_slope_aligned[i] <= 0 or rsi_1d_aligned[i] >= 30)) or
               (signals[i-1] == -0.25 and (ema_1w_slope_aligned[i] >= 0 or rsi_1d_aligned[i] <= 70)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyBias_Volume_Spike"
timeframe = "1d"
leverage = 1.0