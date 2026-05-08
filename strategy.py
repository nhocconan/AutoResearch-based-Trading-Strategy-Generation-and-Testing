#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_HeikinAshi_MACD_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Calculate Heikin-Ashi candles
    ha_close = (open_ + high + low + close) / 4
    ha_open = np.zeros(n)
    ha_open[0] = (open_[0] + close[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))
    
    # Calculate MACD (12,26,9) on HA close
    ha_close_series = pd.Series(ha_close)
    ema_fast = ha_close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = ha_close_series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - signal_line
    macd_hist_values = macd_hist.values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 50-day EMA for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(macd_hist_values[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: MACD histogram positive AND HA close > HA open AND above daily EMA50
            long_cond = (macd_hist_values[i] > 0 and 
                        ha_close[i] > ha_open[i] and
                        close[i] > ema_50_1d_aligned[i] and
                        volume_filter[i])
            
            # Short: MACD histogram negative AND HA close < HA open AND below daily EMA50
            short_cond = (macd_hist_values[i] < 0 and 
                         ha_close[i] < ha_open[i] and
                         close[i] < ema_50_1d_aligned[i] and
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: MACD histogram turns negative OR price crosses below daily EMA50
            if macd_hist_values[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: MACD histogram turns positive OR price crosses above daily EMA50
            if macd_hist_values[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals