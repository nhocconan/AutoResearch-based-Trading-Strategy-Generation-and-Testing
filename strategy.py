#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation with 12h trend filter and 1d volatility regime.
# Long when price > VWAP(20) AND 12h close > 12h EMA50 (uptrend) AND 1d ATR ratio < 0.8 (low vol regime).
# Short when price < VWAP(20) AND 12h close < 12h EMA50 (downtrend) AND 1d ATR ratio < 0.8.
# Exit when price crosses VWAP(20) OR 12h trend reverses OR 1d ATR ratio > 1.2 (high vol breakout).
# Uses 6h timeframe for 50-150 total trades over 4 years. VWAP acts as dynamic support/resistance,
# 12h EMA50 filters trend direction, 1d ATR ratio avoids choppy markets and volatile breakouts.
# Works in bull/bear by trading with 12h trend only in low volatility regimes.

name = "6h_VWAP_12hTrend_1dATRRatio_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volatility regime (ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # < 0.8 = low vol, > 1.2 = high vol
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 6h VWAP(20) - typical price * volume cumulative
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)  # cumulative sum treating NaN as 0
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    # Rolling window VWAP: reset every 20 periods
    vwap_20 = np.full(n, np.nan)
    for i in range(20, n):
        start_idx = i - 19
        window_pv = np.nansum(pv[start_idx:i+1])
        window_vol = np.nansum(volume[start_idx:i+1])
        if window_vol > 0:
            vwap_20[i] = window_pv / window_vol
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(vwap_20[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # 12h trend conditions
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # 1d volatility regime
        low_vol = atr_ratio_aligned[i] < 0.8
        high_vol = atr_ratio_aligned[i] > 1.2
        
        # VWAP deviation
        price_above_vwap = close[i] > vwap_20[i]
        price_below_vwap = close[i] < vwap_20[i]
        
        if position == 0:
            # Long: price > VWAP AND 12h uptrend AND low vol regime
            if price_above_vwap and uptrend and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: price < VWAP AND 12h downtrend AND low vol regime
            elif price_below_vwap and downtrend and low_vol:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < VWAP OR 12h trend reverses to downtrend OR high vol breakout
            if price_below_vwap or not uptrend or high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > VWAP OR 12h trend reverses to uptrend OR high vol breakout
            if price_above_vwap or not downtrend or high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals