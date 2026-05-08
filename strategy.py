#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-period RSI + 4h Bollinger Bands mean reversion + 1d trend filter + session filter (08-20 UTC)
# Mean reversion in range-bound markets: buy when RSI < 30 and price touches lower Bollinger Band (4h) in uptrend (1d)
# Sell when RSI > 70 and price touches upper Bollinger Band (4h) in downtrend (1d)
# Uses 4h for signal direction (BBands + trend), 1h only for entry timing (RSI)
# Session filter reduces noise trades outside active hours
# Target: 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag

name = "1h_RSI4_BBands4h_1dTrend_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # RSI(4) on 1h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    bb_middle = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align Bollinger Bands to 1h
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    
    # Get 1d data for trend filter (EMA50 slope)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_slope = ema50_1d[1:] - ema50_1d[:-1]
    ema50_1d_slope = np.concatenate([[0], ema50_1d_slope])
    ema50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for RSI and indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(ema50_1d_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        bb_upper_val = bb_upper_aligned[i]
        bb_lower_val = bb_lower_aligned[i]
        ema50_slope = ema50_1d_slope_aligned[i]
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if position == 0 and in_session:
            # Enter long: RSI < 30 (oversold) + price at or below lower BB + 1d uptrend
            if rsi_val < 30 and close[i] <= bb_lower_val and ema50_slope > 0:
                signals[i] = 0.20
                position = 1
            # Enter short: RSI > 70 (overbought) + price at or above upper BB + 1d downtrend
            elif rsi_val > 70 and close[i] >= bb_upper_val and ema50_slope < 0:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion) or 1d trend turns down
            if rsi_val > 50 or ema50_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion) or 1d trend turns up
            if rsi_val < 50 or ema50_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals