# 1h_Session_Volatility_Trend_Follow_1dTrend
# Hypothesis: Trend following with volatility filter and session restriction works in both bull and bear markets.
# Uses 1d ADX for trend strength, 1h ATR for volatility, and restricts to 08-20 UTC.
# Position size is 0.20 to manage drawdown. Targets 15-30 trades/year by requiring strong trend + volatility + session.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX (trend strength)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move = np.where(up_move < 0, 0, up_move)
    down_move = np.where(down_move < 0, 0, down_move)
    
    # Smoothed TR, +DM, -DM
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_dm = pd.Series(up_move).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    minus_dm = pd.Series(down_move).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1h ATR(14) for volatility
    tr1h = high - low
    tr2h = np.abs(high - np.roll(close, 1))
    tr3h = np.abs(low - np.roll(close, 1))
    tr_h = np.maximum(tr1h, np.maximum(tr2h, tr3h))
    tr_h[0] = tr1h[0]
    atr_1h = pd.Series(tr_h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h EMA(20) for trend direction
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(atr_1h[i]) or 
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend strength filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volatility filter: ATR > 0.5% of price (avoid choppy low-vol periods)
        vol_filter = atr_1h[i] > (close[i] * 0.005)
        
        # Trend direction: price above/below EMA20
        uptrend = close[i] > ema_20[i]
        downtrend = close[i] < ema_20[i]
        
        # Long conditions: strong trend + uptrend + volatility
        long_condition = strong_trend and uptrend and vol_filter
        
        # Short conditions: strong trend + downtrend + volatility
        short_condition = strong_trend and downtrend and vol_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.20
            position = -1
        # Exit conditions: trend weakens or reverses
        elif position == 1 and (adx_aligned[i] < 20 or downtrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (adx_aligned[i] < 20 or uptrend):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Session_Volatility_Trend_Follow_1dTrend"
timeframe = "1h"
leverage = 1.0