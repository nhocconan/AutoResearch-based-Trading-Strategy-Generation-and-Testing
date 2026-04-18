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
    
    # Get 12h data for trend and volume
    df_12h = get_htf_data(prices, '12h')
    open_12h = df_12h['open'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA(34) for trend
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 4h ATR(14) for stop loss and position sizing
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 34  # need EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        vol_confirmed = volume_12h_aligned[i] > 1.5 * vol_ma_12h_aligned[i]
        
        # Trend filter: price above/below 12h EMA34
        trend_up = close[i] > ema_34_12h_aligned[i]
        trend_down = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long entry: price above 12h open + 0.25*ATR, with volume and trend filter
            if (close[i] > open_12h_aligned[i] + 0.25 * atr_4h[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 12h open - 0.25*ATR, with volume and trend filter
            elif (close[i] < open_12h_aligned[i] - 0.25 * atr_4h[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below 12h open or ATR-based stop
            if close[i] < open_12h_aligned[i] - 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12h open or ATR-based stop
            if close[i] > open_12h_aligned[i] + 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA34_Volume_OpenBreakout"
timeframe = "4h"
leverage = 1.0