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
    
    # Load daily HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate daily ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    dx = np.zeros_like(atr_1d)
    dx[:] = np.nan
    mask = atr_1d > 0
    dx[mask] = 100 * np.abs(dm_plus[mask] - dm_minus[mask]) / (dm_plus[mask] + dm_minus[mask])
    adx_1d = wilder_smooth(dx, 14)
    adx_1d = np.concatenate([np.full(27, np.nan), adx_1d[27:]])  # Account for smoothing delay
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12-period RSI on 12h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/12, adjust=False, min_periods=12).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/12, adjust=False, min_periods=12).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        if np.isnan(adx_1d_aligned[i]) or np.isnan(rsi[i]):
            continue
            
        # Strong trend filter: only trade when ADX > 25
        if adx_1d_aligned[i] > 25:
            # In strong trend, use RSI for pullback entries
            if rsi[i] < 30 and close[i] > close[i-1]:  # Oversold bounce in uptrend
                signals[i] = 0.25
            elif rsi[i] > 70 and close[i] < close[i-1]:  # Overbought rejection in downtrend
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            # In weak trend/ranging market, mean reversion at RSI extremes
            if rsi[i] < 20:  # Deep oversold
                signals[i] = 0.25
            elif rsi[i] > 80:  # Deep overbought
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
    
    return signals

name = "12h_ADX_RSI_Trend_Following"
timeframe = "12h"
leverage = 1.0