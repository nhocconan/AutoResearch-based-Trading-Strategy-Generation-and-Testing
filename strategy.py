#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h volume-weighted RSI + 1w trend filter
    # Long: 6h VWRSI < 30 (oversold) AND 1w close > 1w EMA200 (bullish trend)
    # Short: 6h VWRSI > 70 (overbought) AND 1w close < 1w EMA200 (bearish trend)
    # Exit: VWRSI returns to neutral zone (40-60) OR trend reversal
    # Uses volume weighting to filter weak moves, weekly trend for major direction
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 6h volume-weighted RSI (14-period)
    # Typical price = (H+L+C)/3
    typical_price = (high + low + close) / 3.0
    # Volume-weighted typical price change
    vwtp = typical_price * volume
    # Price change
    delta = np.diff(typical_price, prepend=typical_price[0])
    # Volume-weighted gain/loss
    gain = np.where(delta > 0, delta * volume, 0.0)
    loss = np.where(delta < 0, -delta * volume, 0.0)
    
    # Smoothed average gain/loss (Wilder's smoothing)
    def wilders_smoothing(data, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    vwrsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vwrsi[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA200
        bullish_trend = close[i] > ema_200_1w_aligned[i]
        bearish_trend = close[i] < ema_200_1w_aligned[i]
        
        # VWRSI signals
        oversold = vwrsi[i] < 30
        overbought = vwrsi[i] > 70
        neutral_exit = (vwrsi[i] >= 40) and (vwrsi[i] <= 60)
        
        # Entry logic: VWRSI extreme + trend alignment
        long_entry = oversold and bullish_trend
        short_entry = overbought and bearish_trend
        
        # Exit logic: VWRSI returns to neutral OR trend reversal
        long_exit = neutral_exit or (not bullish_trend)
        short_exit = neutral_exit or (not bearish_trend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_vwrsi_trend_filter_v1"
timeframe = "6h"
leverage = 1.0