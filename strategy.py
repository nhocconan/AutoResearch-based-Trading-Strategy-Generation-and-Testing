#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Volume-Weighted RSI with 4h Trend Filter and Session Hours
# Long when: 4h EMA50 uptrend AND 1h VW-RSI < 30 (oversold) during 08-20 UTC
# Short when: 4h EMA50 downtrend AND 1h VW-RSI > 70 (overbought) during 08-20 UTC
# Uses volume-weighted RSI for better mean reversion signals, 4h EMA for trend filter,
# session filter to reduce noise, and discrete sizing (0.20) to minimize fee churn.
# Works in bull/bear via trend filter + mean reversion at extremes.
# Timeframe: 1h (primary), HTF: 4h for trend.

name = "1h_VolWeightedRSI_4hEMA50_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop for 4h EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Volume-Weighted RSI (14-period)
    # Typical price = (H+L+C)/3
    typical_price = (high + low + close) / 3.0
    # Volume-weighted typical price
    vwtp = typical_price * volume
    
    # Calculate changes
    delta = pd.Series(typical_price).diff().values
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Volume-weighted gains and losses
    vt_gain = gain * volume
    vt_loss = loss * volume
    
    # Smoothed volume-weighted RSI
    avg_vt_gain = pd.Series(vt_gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_vt_loss = pd.Series(vt_loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_vt_loss != 0, avg_vt_gain / avg_vt_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_vw_rsi = vw_rsi[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. 4h EMA50 turns down (trend change)
            # 2. VW-RSI > 50 (mean reversion exit)
            if (curr_close < curr_ema_50_4h or curr_vw_rsi > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. 4h EMA50 turns up (trend change)
            # 2. VW-RSI < 50 (mean reversion exit)
            if (curr_close > curr_ema_50_4h or curr_vw_rsi < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: 4h UPTREND AND VW-RSI oversold (<30)
            if (curr_close > curr_ema_50_4h) and (curr_vw_rsi < 30):
                signals[i] = 0.20
                position = 1
            # Short entry: 4h DOWNTREND AND VW-RSI overbought (>70)
            elif (curr_close < curr_ema_50_4h) and (curr_vw_rsi > 70):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals