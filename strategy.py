#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(14) mean reversion with 4-hour ADX(14) trend filter and session filter (08-20 UTC)
# Long when RSI < 30 (oversold) + 4h ADX < 25 (weak trend/range) + within active session
# Short when RSI > 70 (overbought) + 4h ADX < 25 (weak trend/range) + within active session
# Exit when RSI crosses back to 50 (mean reversion complete)
# Stoploss at 2.0 * ATR(14) on 1h timeframe
# Position size: 0.20 (20% of capital)
# Uses 4h ADX for regime detection to avoid trending markets where mean reversion fails
# Target: 100-200 total trades over 4 years (25-50/year) with session filter reducing noise

name = "1h_rsi14_4h_adx14_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4-hour ADX (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_4h, prepend=high_4h[0])
    down_move = np.diff(low_4h, prepend=low_4h[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # RSI(14) on 1-hour timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for stoploss on 1-hour timeframe
    tr1_1h = high - low
    tr2_1h = np.abs(high - np.roll(close, 1))
    tr3_1h = np.abs(low - np.roll(close, 1))
    tr2_1h[0] = tr1_1h[0]
    tr3_1h[0] = tr1_1h[0]
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    atr = pd.Series(tr_1h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08:00-20:00 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses back to 50 (mean reversion complete)
            elif rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses back to 50 (mean reversion complete)
            elif rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme + weak trend (ADX < 25) + session filter
            rsi_oversold = rsi[i] < 30
            rsi_overbought = rsi[i] > 70
            weak_trend = adx_aligned[i] < 25
            session_ok = in_session[i]
            
            # Long: RSI oversold + weak trend + session
            if rsi_oversold and weak_trend and session_ok:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI overbought + weak trend + session
            elif rsi_overbought and weak_trend and session_ok:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals