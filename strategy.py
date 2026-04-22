#!/usr/bin/env python3

"""
Hypothesis: 4-hour Relative Strength Index (RSI) with 1-day volume-weighted average price (VWAP) trend filter and 1-day volatility regime filter.
RSI provides mean-reversion signals at extremes while VWAP trend filter ensures trades align with the daily institutional flow.
Volatility regime filter (using ATR ratio) avoids choppy markets where mean reversion fails.
This combination should work in both bull and bear markets by adapting to daily trend and volatility conditions.
Target: 20-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily VWAP for trend filter
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_1d = (np.cumsum(typical_price_1d * df_1d['volume'].values) / 
               np.cumsum(df_1d['volume'].values))
    
    # Calculate daily ATR for volatility regime filter
    high_low_1d = df_1d['high'].values - df_1d['low'].values
    high_close_1d = np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))
    low_close_1d = np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))
    true_range_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_1d = pd.Series(true_range_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4-period RSI for 4h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=4, min_periods=4).mean().values
    avg_loss = pd.Series(loss).rolling(window=4, min_periods=4).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily indicators to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h ATR for volatility regime (current volatility)
    high_low = high - low
    high_close = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_4h = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Volatility regime: current 4h ATR vs daily ATR (normalized)
    vol_regime = atr_4h / atr_aligned  # < 1 = low vol, > 1 = high vol
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: trade only in low to normal volatility (avoid extreme volatility)
        # vol_regime between 0.5 and 2.0 (ATR ratio)
        if vol_regime[i] < 0.5 or vol_regime[i] > 2.0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (< 30) AND price below VWAP (mean reversion to VWAP)
            if (rsi[i] < 30 and close[i] < vwap_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (> 70) AND price above VWAP (mean reversion to VWAP)
            elif (rsi[i] > 70 and close[i] > vwap_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI returns to neutral zone (40-60) OR price crosses VWAP
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI >= 40 OR price crosses above VWAP
                if (rsi[i] >= 40 or close[i] >= vwap_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI <= 60 OR price crosses below VWAP
                if (rsi[i] <= 60 or close[i] <= vwap_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI_VWAP_MeanReversion_VolRegime"
timeframe = "4h"
leverage = 1.0