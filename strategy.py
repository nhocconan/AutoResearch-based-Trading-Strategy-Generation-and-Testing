#!/usr/bin/env python3
# 4h_VWAP_Pullback_Trend_Filter
# Hypothesis: Price pulling back to VWAP during strong trends offers high-probability entries.
# Uses 1d trend filter (EMA50) to align with higher timeframe momentum.
# Long when: price > 1d EMA50, pulls back to VWAP (within 0.5%), and shows bullish momentum (close > open).
# Short when: price < 1d EMA50, pulls back to VWAP (within 0.5%), and shows bearish momentum (close < open).
# VWAP calculated intraday (4h bars) and reset daily. Designed for 4h timeframe to reduce trade frequency.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell pullbacks in downtrend).

name = "4h_VWAP_Pullback_Trend_Filter"
timeframe = "4h"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Intraday VWAP (reset daily)
    # Approximate VWAP using typical price * volume
    typical_price = (high + low + close) / 3.0
    vwap_num = typical_price * volume
    vwap_den = volume
    
    # Get unique dates from open_time
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    vwap = np.zeros(n)
    for date in unique_dates:
        mask = (dates == date)
        if not np.any(mask):
            continue
        # Cumulative sum within the day
        cum_vwap_num = np.cumsum(vwap_num[mask])
        cum_vwap_den = np.cumsum(vwap_den[mask])
        # Avoid division by zero
        vwap_day = np.divide(cum_vwap_num, cum_vwap_den, out=np.zeros_like(cum_vwap_num), where=cum_vwap_den!=0)
        vwap[mask] = vwap_day
    
    # VWAP deviation band (0.5%)
    vwap_upper = vwap * 1.005
    vwap_lower = vwap * 0.995
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and VWAP stability
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vwap[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Pullback to VWAP condition (within 0.5% of VWAP)
        near_vwap = (low[i] <= vwap_upper[i]) and (high[i] >= vwap_lower[i])
        
        if position == 0:
            # Long: uptrend, pullback to VWAP, bullish candle
            if (close[i] > ema_50_1d_aligned[i] and 
                near_vwap and 
                close[i] > open_price[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend, pullback to VWAP, bearish candle
            elif (close[i] < ema_50_1d_aligned[i] and 
                  near_vwap and 
                  close[i] < open_price[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or price moves away from VWAP
            if position == 1:
                # Exit long: trend turns down OR price moves above VWAP upper band
                if (close[i] < ema_50_1d_aligned[i]) or (close[i] > vwap_upper[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR price moves below VWAP lower band
                if (close[i] > ema_50_1d_aligned[i]) or (close[i] < vwap_lower[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals