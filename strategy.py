#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(21) for trend
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 12h ATR for entry and stop
    tr_12h_1 = high - low
    tr_12h_2 = np.abs(high - np.roll(close, 1))
    tr_12h_3 = np.abs(low - np.roll(close, 1))
    tr_12h_1[0] = high[0] - low[0]
    tr_12h_2[0] = np.abs(high[0] - close[0])
    tr_12h_3[0] = np.abs(low[0] - close[0])
    tr_12h = np.maximum(tr_12h_1, np.maximum(tr_12h_2, tr_12h_3))
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21)  # need volume MA and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA21
        trend_up = close[i] > ema_21_1w_aligned[i]
        trend_down = close[i] < ema_21_1w_aligned[i]
        
        if position == 0:
            # Long entry: price above 12h open + 0.25*ATR, with volume and trend filter
            if (close[i] > open_price[i] + 0.25 * atr_12h[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 12h open - 0.25*ATR, with volume and trend filter
            elif (close[i] < open_price[i] - 0.25 * atr_12h[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below 12h open or ATR-based stop
            if close[i] < open_price[i] - 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h open or ATR-based stop
            if close[i] > open_price[i] + 1.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA21_Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0