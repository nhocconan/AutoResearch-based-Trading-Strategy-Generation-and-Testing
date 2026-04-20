#!/usr/bin/env python3
# 1d_1w_HMA_Trend_Follow
# Hypothesis: On 1d timeframe, trade in direction of 1-week HMA(21) trend with volume confirmation and ADX filter.
# Uses 1d price action for entry timing: buy pullbacks to 1d VWAP in uptrend, sell rallies to 1d VWAP in downtrend.
# Targets 15-25 trades/year by requiring trend alignment + volume + VWAP touch.
# Works in bull markets via trend following and bear markets via short side.

name = "1d_1w_HMA_Trend_Follow"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w HMA(21) - Hull Moving Average
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, window):
        half = window // 2
        sqrt = int(np.sqrt(window))
        wma_half = wma(values, half)
        wma_full = wma(values, window)
        wma_half = np.concatenate([np.full(len(values) - len(wma_half), np.nan), wma_half])
        wma_full = np.concatenate([np.full(len(values) - len(wma_full), np.nan), wma_full])
        diff = 2 * wma_half - wma_full
        return wma(diff, sqrt)
    
    close_1w = df_1w['close'].values
    hma_1w = hma(close_1w, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d VWAP
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, np.nan)
    
    # Calculate 1d ADX(14) for trend strength filter
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Uptrend: price above weekly HMA
            if close[i] > hma_1w_aligned[i]:
                # Long on pullback to VWAP with volume confirmation
                if (close[i] <= vwap[i] * 1.01 and 
                    close[i] >= vwap[i] * 0.99 and
                    volume[i] > 1.5 * vol_ma[i] and
                    adx[i] > 20):
                    signals[i] = 0.25
                    position = 1
            # Downtrend: price below weekly HMA
            elif close[i] < hma_1w_aligned[i]:
                # Short on rally to VWAP with volume confirmation
                if (close[i] >= vwap[i] * 0.99 and 
                    close[i] <= vwap[i] * 1.01 and
                    volume[i] > 1.5 * vol_ma[i] and
                    adx[i] > 20):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly HMA or reaches 2% above VWAP
            if close[i] < hma_1w_aligned[i] * 0.995 or close[i] > vwap[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly HMA or reaches 2% below VWAP
            if close[i] > hma_1w_aligned[i] * 1.005 or close[i] < vwap[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals