#!/usr/bin/env python3
"""
6h_1d_200EMA_WeeklyTrend_StochRSI_Oversold
Hypothesis: On 6H timeframe, buy when price > daily 200 EMA (long-term uptrend), weekly trend is up (price > weekly 200 EMA), and StochRSI is oversold (<0.2). Sell when StochRSI overbought (>0.8) or trend breaks. Designed to catch pullbacks in strong uptrends, works in bull markets by buying dips, avoids bear markets by requiring both daily and weekly uptrend. Low turnover expected (~15-25 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily and weekly data once
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 200 or len(df_1w) < 200:
        return np.zeros(n)
    
    # Daily 200 EMA
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Weekly 200 EMA
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # StochRSI on 6H close (14-period RSI, then Stoch of RSI)
    close = prices['close'].values
    rsi_period = 14
    stoch_period = 14
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    avg_gain = wilder_smooth(gain, rsi_period)
    avg_loss = wilder_smooth(loss, rsi_period)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_min = np.full_like(rsi, np.nan)
    rsi_max = np.full_like(rsi, np.nan)
    for i in range(stoch_period-1, len(rsi)):
        rsi_min[i] = np.nanmin(rsi[i-stoch_period+1:i+1])
        rsi_max[i] = np.nanmax(rsi[i-stoch_period+1:i+1])
    rsi_range = rsi_max - rsi_min
    stoch_rsi = np.where(rsi_range != 0, (rsi - rsi_min) / rsi_range, 0.5)
    
    # Align trends
    trend_daily = ema_200_1d_aligned > 0  # placeholder for actual comparison
    trend_weekly = ema_200_1w_aligned > 0
    
    # Actual trend conditions: price > EMA200
    trend_daily = close > ema_200_1d_aligned
    trend_weekly = close > ema_200_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(stoch_rsi[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price above both daily and weekly 200 EMA, and StochRSI oversold
            if trend_daily[i] and trend_weekly[i] and stoch_rsi[i] < 0.2:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit: StochRSI overbought OR trend breaks (price below either EMA)
            if stoch_rsi[i] > 0.8 or not (trend_daily[i] and trend_weekly[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "6h_1d_200EMA_WeeklyTrend_StochRSI_Oversold"
timeframe = "6h"
leverage = 1.0