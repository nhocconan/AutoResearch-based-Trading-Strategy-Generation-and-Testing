#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND Elder Bull Power > 0 AND price > 1d EMA50
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND price < 1d EMA50
# ATR-based stoploss: exit when price moves against position by 2.0 * ATR(14).
# Alligator catches trends, Elder Ray confirms momentum, 1d EMA filters counter-trend trades.
# Works in bull via Alligator longs, in bear via Alligator shorts.

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Williams Alligator (13,8,5 smoothed with 8,5,3)
    # Jaw: 13-period SMMA smoothed by 8
    # Teeth: 8-period SMMA smoothed by 5
    # Lips: 5-period SMMA smoothed by 3
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(smma(close, 13), 8)
    teeth = smma(smma(close, 8), 5)
    lips = smma(smma(close, 5), 3)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1d EMA(50) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(13*2, 8*2, 5*2, 13, 50)  # warmup for Alligator and EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator: jaws < teeth < lips (alligator sleeping -> waking up bullish)
            bullish_alligator = (curr_jaw < curr_teeth) and (curr_teeth < curr_lips)
            # Bearish Alligator: jaws > teeth > lips (alligator sleeping -> waking up bearish)
            bearish_alligator = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips)
            
            # Bullish entry: bullish Alligator AND positive Bull Power AND price > 1d EMA50
            if bullish_alligator and (bull_power[i] > 0) and (curr_close > curr_ema_50_1d):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Bearish entry: bearish Alligator AND negative Bear Power AND price < 1d EMA50
            elif bearish_alligator and (bear_power[i] < 0) and (curr_close < curr_ema_50_1d):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # ATR-based stoploss: exit when price drops below entry - 2.0 * ATR
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # ATR-based stoploss: exit when price rises above entry + 2.0 * ATR
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals