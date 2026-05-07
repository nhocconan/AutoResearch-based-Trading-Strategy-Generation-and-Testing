# 12h_1dCandlestickPattern_1dTrend_Volume_Confirm
# Detects bullish/bearish engulfing patterns on daily chart with weekly trend filter.
# Long on bullish engulfing in weekly uptrend, short on bearish engulfing in weekly downtrend.
# Engulfing patterns signal reversal, effective in both bull and bear markets.

name = "12h_1dCandlestickPattern_1dTrend_Volume_Confirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for candlestick patterns
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily bullish and bearish engulfing patterns
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    bullish_engulfing = (close_1d > open_1d) & (open_1d.shift(1) > close_1d.shift(1)) & \
                        (close_1d >= open_1d.shift(1)) & (open_1d <= close_1d.shift(1))
    bearish_engulfing = (close_1d < open_1d) & (open_1d.shift(1) < close_1d.shift(1)) & \
                        (close_1d <= open_1d.shift(1)) & (open_1d >= close_1d.shift(1))
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume filter (20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Align daily patterns and weekly EMA to 12h timeframe
    bullish_engulfing_12h = align_htf_to_ltf(prices, df_1d, bullish_engulfing.astype(float))
    bearish_engulfing_12h = align_htf_to_ltf(prices, df_1d, bearish_engulfing.astype(float))
    ema_34_12h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(60, n):
        if (np.isnan(bullish_engulfing_12h[i]) or np.isnan(bearish_engulfing_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish engulfing in weekly uptrend with volume
            if bullish_engulfing_12h[i] and close[i] > ema_34_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing in weekly downtrend with volume
            elif bearish_engulfing_12h[i] and close[i] < ema_34_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish engulfing or price below weekly EMA
            if bearish_engulfing_12h[i] or close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish engulfing or price above weekly EMA
            if bullish_engulfing_12h[i] or close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals