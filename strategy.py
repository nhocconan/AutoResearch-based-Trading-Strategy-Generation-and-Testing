#!/usr/bin/env python3
"""
1h_HTF_Regime_Filter_LongOnly
Hypothesis: On 1h timeframe, use 4h EMA20 trend and 1d ADX regime filter to take longs only in bullish/low-volatility regimes.
Avoids bear markets and choppy conditions by requiring: 1) price > 4h EMA20 (uptrend), 2) 1d ADX < 25 (low trend strength = range/accumulation), 3) 1h close > 1h open (bullish candle).
Designed for low turnover (~15-30 trades/year) by combining tight HTF regime filters with precise 1h entry timing.
Works in bull via trend continuation, in bear via avoidance (flat), and in range via mean-reversion long bias at support.
"""

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
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for EMA20 trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d data for ADX regime filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX(14) on 1d
    plus_dm = np.where((df_1d['high'].diff()) > (df_1d['low'].diff().abs()), np.maximum(df_1d['high'].diff(), 0), 0)
    minus_dm = np.where((df_1d['low'].diff().abs()) > (df_1d['high'].diff()), np.maximum(df_1d['low'].diff().abs(), 0), 0)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Align ADX to 1h timeframe (low ADX = ranging/accumulation regime)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    entry_price = 0.0
    
    # Start index: need enough for 4h EMA20 (20) and 1d ADX (14+14=28)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        
        # Regime filters: 4h uptrend + 1d low ADX (range/accumulation)
        uptrend_4h = curr_close > ema_20_4h_aligned[i]
        low_adx_regime = adx_1d_aligned[i] < 25  # ADX < 25 = weak trend = range/accumulation
        bullish_candle = curr_close > curr_open  # 1h bullish candle
        
        if position == 0:
            # Enter long only in bullish regime with bullish candle
            if uptrend_4h and low_adx_regime and bullish_candle:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when regime deteriorates or trend breaks
            if not uptrend_4h or adx_1d_aligned[i] >= 30:  # ADX >= 30 = strong trend (could be bear)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
    
    return signals

name = "1h_HTF_Regime_Filter_LongOnly"
timeframe = "1h"
leverage = 1.0