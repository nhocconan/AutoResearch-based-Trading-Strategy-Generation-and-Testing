#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h EMA34 trend and 1h RSI divergence for entry timing.
# Uses 4h EMA34 for trend direction and 1h RSI(14) with bullish/bearish divergence for entries.
# Enters only during 08-20 UTC session to avoid low-volume noise.
# Targets 15-37 trades/year (60-150 total over 4 years) with strict entry conditions.
# Works in bull/bear by following higher timeframe trends.
name = "1h_4h_EMA34_RSI_Divergence"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA34 trend (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # RSI(14) for divergence detection
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = (low[i] < low[i-5]) and (rsi[i] > rsi[i-5])
            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = (high[i] > high[i-5]) and (rsi[i] < rsi[i-5])
            
            # Long: above 4h EMA34 AND bullish divergence
            if (close[i] > ema_34_4h_aligned[i] and 
                bullish_div and 
                rsi[i] < 40):  # Oversold condition
                signals[i] = 0.20
                position = 1
            # Short: below 4h EMA34 AND bearish divergence
            elif (close[i] < ema_34_4h_aligned[i] and 
                  bearish_div and 
                  rsi[i] > 60):  # Overbought condition
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 4h EMA34 or RSI overbought
            if close[i] < ema_34_4h_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above 4h EMA34 or RSI oversold
            if close[i] > ema_34_4h_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals