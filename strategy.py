#!/usr/bin/env python3
# 1h_ADX_Trend_Follow_4h1dTrend_Filter
# Hypothesis: Use ADX(14) > 25 to detect trending markets on 1h, with 4h/1d trend filters (price > EMA50/200).
# Long when ADX > 25, price > 4h EMA50, and price > 1d EMA200.
# Short when ADX > 25, price < 4h EMA50, and price < 1d EMA200.
# Exit when ADX falls below 20 or trend alignment breaks.
# Uses 4h/1d for signal direction, 1h only for entry timing via ADX.
# Targets 15-30 trades/year via strict ADX + trend alignment filters.

name = "1h_ADX_Trend_Follow_4h1dTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ADX on 1h data
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    period = 14
    atr[period] = np.nansum(tr[1:period+1])
    plus_dm_sum = np.nansum(plus_dm[1:period+1])
    minus_dm_sum = np.nansum(minus_dm[1:period+1])
    
    for i in range(period+1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sum = (plus_dm_sum * (period-1) + plus_dm[i]) / period
        minus_dm_sum = (minus_dm_sum * (period-1) + minus_dm[i]) / period
        
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        if i < 2*period:
            adx[i] = np.nan
        else:
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period if not np.isnan(adx[i-1]) else np.nansum(dx[period+1:i+1]) / (i - period)
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate EMAs on HTF data
    ema_4h_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 1h timeframe
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 2*period)  # Ensure ADX and indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(adx[i]) or np.isnan(ema_4h_50_aligned[i]) or 
            np.isnan(ema_1d_200_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        adx_val = adx[i]
        ema_4h_val = ema_4h_50_aligned[i]
        ema_1d_val = ema_1d_200_aligned[i]
        close_val = close[i]
        
        # Determine trend alignment
        uptrend = close_val > ema_4h_val and close_val > ema_1d_val
        downtrend = close_val < ema_4h_val and close_val < ema_1d_val
        
        if position == 0:
            # ENTER LONG: Strong uptrend with ADX > 25
            if adx_val > 25 and uptrend:
                signals[i] = 0.25
                position = 1
            # ENTER SHORT: Strong downtrend with ADX > 25
            elif adx_val > 25 and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakening or ADX dropping
            if adx_val < 20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakening or ADX dropping
            if adx_val < 20 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals