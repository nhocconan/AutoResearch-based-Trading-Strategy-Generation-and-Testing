#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and session filter.
Long when 1h RSI < 30 AND 4h EMA50 > EMA200 (uptrend) AND hour between 08-20 UTC.
Short when 1h RSI > 70 AND 4h EMA50 < EMA200 (downtrend) AND hour between 08-20 UTC.
Exit when 1h RSI crosses 50 (mean reversion completion) OR hour outside 08-20 UTC.
Uses 4h for trend direction (reduces whipsaw) and 1h for precise entry/exit timing.
Session filter avoids low-liquidity Asian session noise. Target: 60-150 trades over 4 years.
"""

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
    
    # Pre-compute session hours for filter
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMAs on 4h timeframe
    close_4h_series = pd.Series(close_4h)
    ema_50 = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_4h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_50_val = ema_50_aligned[i]
        ema_200_val = ema_200_aligned[i]
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND 4h uptrend (EMA50 > EMA200) AND in session
            if rsi_val < 30 and ema_50_val > ema_200_val and in_session:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) AND 4h downtrend (EMA50 < EMA200) AND in session
            elif rsi_val > 70 and ema_50_val < ema_200_val and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion) OR outside session
            if rsi_val > 50 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion) OR outside session
            if rsi_val < 50 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMATrend_SessionFilter"
timeframe = "1h"
leverage = 1.0