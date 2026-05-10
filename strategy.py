# 1D_Pullback_to_200SMA_with_200MA_Volume_Spike
# Hypothesis: In both bull and bear markets, strong trends often retrace to the 200-day SMA before resuming.
# Long entries occur when price pulls back to the 200SMA with a bullish candle, volume spike, and price above 200-day EMA (trend filter).
# Short entries occur when price rallies to the 200SMA with a bearish candle, volume spike, and price below 200-day EMA.
# The 200SMA acts as dynamic support/resistance; the 200EMA confirms the trend direction; volume spike confirms institutional interest.
# Designed for low trade frequency (5-15/year) to minimize fee drag.

name = "1D_Pullback_to_200SMA_with_200MA_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend context
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 200-day SMA and EMA on daily data
    def sma(arr, window):
        res = np.full_like(arr, np.nan)
        if len(arr) >= window:
            for i in range(window-1, len(arr)):
                res[i] = np.mean(arr[i-window+1:i+1])
        return res
    
    sma_200 = sma(close, 200)
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 20-day volume average for spike detection
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need full 200-day history
    
    for i in range(start_idx, n):
        if np.isnan(sma_200[i]) or np.isnan(ema_200[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-day average
        volume_spike = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Bullish candle: close > open
        bullish_candle = close[i] > prices['open'].iloc[i]
        # Bearish candle: close < open
        bearish_candle = close[i] < prices['open'].iloc[i]
        
        # Price near 200SMA (within 1%)
        price_near_sma = abs(close[i] - sma_200[i]) / sma_200[i] < 0.01
        
        if position == 0:
            # Long: price pulls back to 200SMA with bullish candle, volume spike, and above 200EMA (uptrend)
            if price_near_sma and bullish_candle and volume_spike and close[i] > ema_200[i]:
                signals[i] = 0.25
                position = 1
            # Short: price rallies to 200SMA with bearish candle, volume spike, and below 200EMA (downtrend)
            elif price_near_sma and bearish_candle and volume_spike and close[i] < ema_200[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 200EMA OR weekly EMA50 turns down
            if close[i] < ema_200[i] or ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 200EMA OR weekly EMA50 turns up
            if close[i] > ema_200[i] or ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals