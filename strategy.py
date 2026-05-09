#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI(14) mean reversion with 1w trend filter and volume confirmation.
# RSI(14) < 30 for long entry in 1w uptrend, RSI(14) > 70 for short entry in 1w downtrend.
# Volume must be > 1.5x 20-period EMA to confirm momentum.
# Exit when RSI returns to neutral (40-60) or trend reverses.
# Designed for 1d timeframe to target 7-25 trades/year, avoiding fee drag.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
name = "1d_RSI14_MeanRev_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Need RSI warmup
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) + 1w uptrend + volume confirmation
            if (rsi[i] < 30 and price > ema_34_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 (overbought) + 1w downtrend + volume confirmation
            elif (rsi[i] > 70 and price < ema_34_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or trend reverses
            if rsi[i] > 40 or price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or trend reverses
            if rsi[i] < 60 or price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals