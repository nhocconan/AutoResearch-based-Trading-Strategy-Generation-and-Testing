#!/usr/bin/env python3
"""
1h_MultiTF_Trend_Filter_V1
1h strategy with 4h trend filter and 1d momentum confirmation.
- Long: 4h EMA21 > EMA50 + 1d RSI < 30 (oversold) + price > 1h VWAP
- Short: 4h EMA21 < EMA50 + 1d RSI > 70 (overbought) + price < 1h VWAP
- Exit: Trend reversal or RSI mean reversion (RSI > 50 for longs, RSI < 50 for shorts)
- Session filter: 08-20 UTC only
Designed for ~15-30 trades/year per symbol (60-120 total over 4 years)
Works in bull markets (trend continuation on pullbacks) and bear markets (mean reversion in trends)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA21 and EMA50 for trend
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for momentum (RSI)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1h VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / (vwap_den + 1e-10)
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for 4h EMA50 and 1d RSI
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions from 4h
        uptrend_4h = ema_21_4h_aligned[i] > ema_50_4h_aligned[i]
        downtrend_4h = ema_21_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Momentum conditions from 1d RSI
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        rsi_long_exit = rsi_1d_aligned[i] > 50  # exit long when RSI > 50
        rsi_short_exit = rsi_1d_aligned[i] < 50  # exit short when RSI < 50
        
        # 1h price vs VWAP for entry timing
        price_above_vwap = close[i] > vwap[i]
        price_below_vwap = close[i] < vwap[i]
        
        if position == 0:
            # Long: 4h uptrend + 1d oversold + price above VWAP
            if uptrend_4h and rsi_oversold and price_above_vwap:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + 1d overbought + price below VWAP
            elif downtrend_4h and rsi_overbought and price_below_vwap:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h trend reversal OR 1d RSI mean reversion (>50) OR price below VWAP
            if (not uptrend_4h) or rsi_long_exit or (not price_above_vwap):
                signals[i] = 0.0  # flat
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend reversal OR 1d RSI mean reversion (<50) OR price above VWAP
            if (not downtrend_4h) or rsi_short_exit or (not price_below_vwap):
                signals[i] = 0.0  # flat
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_MultiTF_Trend_Filter_V1"
timeframe = "1h"
leverage = 1.0