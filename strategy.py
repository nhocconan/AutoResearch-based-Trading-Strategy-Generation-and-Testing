#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d trend filter with RSI pullback
# Uses 4h EMA for trend direction, 1d ADX for trend strength, and 1h RSI for pullback entry.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Timeframe: 1h, HTF: 4h/1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Load 1d data for ADX trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA(50)
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d ADX(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(close_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(close_1d, 1)), 
                        np.maximum(np.roll(close_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            continue
        
        # Check session
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Long: uptrend (price > 4h EMA) + strong trend (ADX > 25) + RSI pullback (< 40)
        if (close[i] > ema_4h_aligned[i] and
            adx_1d_aligned[i] > 25 and
            rsi[i] < 40 and
            in_session and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: downtrend (price < 4h EMA) + strong trend (ADX > 25) + RSI pullback (> 60)
        elif (close[i] < ema_4h_aligned[i] and
              adx_1d_aligned[i] > 25 and
              rsi[i] > 60 and
              in_session and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend weakening (ADX < 20) or opposite RSI extreme
        elif position == 1 and (adx_1d_aligned[i] < 20 or rsi[i] > 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx_1d_aligned[i] < 20 or rsi[i] < 30):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_4h1d_EMA_ADX_RSI_Pullback"
timeframe = "1h"
leverage = 1.0