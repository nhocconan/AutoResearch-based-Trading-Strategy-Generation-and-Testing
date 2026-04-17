#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4-period ATR for breakout sensitivity
    tr_l = np.abs(high - low)
    tr_h = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
    tr_lc = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
    tr_l = np.concatenate([[np.nan], tr_l[1:]])
    tr_h = np.concatenate([[np.nan], tr_h[1:]])
    tr_lc = np.concatenate([[np.nan], tr_lc[1:]])
    tr_combined = np.maximum(tr_l, np.maximum(tr_h, tr_lc))
    atr_4 = pd.Series(tr_combined).rolling(window=4, min_periods=4).mean().values
    
    # Get weekly data for structural trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly EMA50, daily ATR14
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_1h[i]) or 
            np.isnan(atr_4[i]) or 
            np.isnan(ema50_1h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is elevated (ATR14 > 50th percentile of last 50 periods)
        if i >= 50:
            vol_window = atr_14_1h[i-50:i]
            vol_median = np.nanmedian(vol_window)
            vol_filter = atr_14_1h[i] > vol_median
        else:
            vol_filter = False
        
        # Breakout condition: price moves more than 0.5 * ATR(4) from open
        breakout_up = (close[i] - prices['open'].values[i]) > (0.5 * atr_4[i])
        breakout_down = (prices['open'].values[i] - close[i]) > (0.5 * atr_4[i])
        
        # Trend filter: weekly EMA50 direction
        if i >= 1:
            ema_prev = ema50_1h[i-1]
            ema_curr = ema50_1h[i]
            trend_up = ema_curr > ema_prev
            trend_down = ema_curr < ema_prev
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long: upward breakout during up-trend with elevated volatility
            if (breakout_up and trend_up and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: downward breakout during down-trend with elevated volatility
            elif (breakout_down and trend_down and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: downward breakout or trend change
            if (breakout_down or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: upward breakout or trend change
            if (breakout_up or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Volatility_Breakout_EMA50_Trend"
timeframe = "4h"
leverage = 1.0