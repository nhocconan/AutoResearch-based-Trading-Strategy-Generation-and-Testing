#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and 1d volatility filter
# RSI < 30 (oversold) in uptrend (price > 4h EMA20) for long entries
# RSI > 70 (overbought) in downtrend (price < 4h EMA20) for short entries
# 1d ATR ratio filter to avoid low volatility chop
# Session filter: 08-20 UTC to avoid low volume periods
# Position size: 0.20 (20%)
# Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 20-period EMA on 4h timeframe for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Load 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d timeframe
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR is undefined
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-period SMA of ATR for volatility regime
    atr_ma50 = pd.Series(atr14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr14_1d / atr_ma50  # >1 = high volatility, <1 = low volatility
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate RSI(14) on 1h price data
    delta = pd.Series(close_1d).diff().values  # This is wrong - need to use 1h data
    # Fix: calculate RSI on actual 1h close prices
    close = prices['close'].values
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])  # prepend 0 for first element
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.2x 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.2)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema20_4h_aligned[i]) or np.isnan(rsi[i]) or \
           np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply filters
        in_session = session_filter[i]
        has_volume = vol_filter[i]
        high_vol = atr_ratio_aligned[i] > 1.2  # Only trade in elevated volatility
        
        price = close[i]
        rsi_val = rsi[i]
        is_uptrend = price > ema20_4h_aligned[i]
        is_downtrend = price < ema20_4h_aligned[i]
        
        if position == 0:
            # Long entry: RSI < 30 (oversold) + uptrend + high vol + session + volume
            long_signal = (rsi_val < 30) and is_uptrend and high_vol and in_session and has_volume
            
            # Short entry: RSI > 70 (overbought) + downtrend + high vol + session + volume
            short_signal = (rsi_val > 70) and is_downtrend and high_vol and in_session and has_volume
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion) or trend change
            if rsi_val > 50 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion) or trend change
            if rsi_val < 50 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hTrend_1dVolFilter_Session"
timeframe = "1h"
leverage = 1.0