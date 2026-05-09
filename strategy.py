#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h RSI mean-reversion with 1d trend filter and session filter
# Long when 4h RSI < 30 (oversold) and 1d close > 1d EMA50 (bullish trend) during active session (08-20 UTC)
# Short when 4h RSI > 70 (overbought) and 1d close < 1d EMA50 (bearish trend) during active session
# Exit when 4h RSI crosses back above 50 (long exit) or below 50 (short exit)
# Position size: 0.20 to limit drawdown and reduce trade frequency
# Designed to work in both bull (trend filter) and bear (mean reversion in trends) markets

name = "1h_RSI_MeanReversion_1dTrend_Session"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h RSI for mean reversion signals (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Get 4h data for RSI calculation (source of truth for signal)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 15:
        return np.zeros(n)
    
    # Calculate 4h RSI (14-period)
    close_4h = df_4h['close'].values
    delta_4h = pd.Series(close_4h).diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_4h = avg_gain_4h / avg_loss_4h
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_values = rsi_4h.values
    
    # Align 4h RSI to 1h timeframe (waits for 4h bar close)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_values)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe (waits for daily close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 4h RSI oversold (<30) AND 1d trend bullish (close > EMA50)
            if (rsi_4h_aligned[i] < 30 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h RSI overbought (>70) AND 1d trend bearish (close < EMA50)
            elif (rsi_4h_aligned[i] > 70 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h RSI crosses back above 50 (mean reversion complete)
            if rsi_4h_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h RSI crosses back below 50 (mean reversion complete)
            if rsi_4h_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals