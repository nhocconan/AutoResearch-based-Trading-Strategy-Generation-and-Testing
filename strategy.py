#!/usr/bin/env python3
"""
12h_1d_vwap_bounce_v1
Strategy: 12h VWAP bounce with 1-day trend filter and volume confirmation
Timeframe: 12h
Leverage: 1.0
Hypothesis: Price tends to revert to VWAP during low volatility periods when the 1-day trend is aligned. Uses 12h price crossing above/below VWAP with 1-day EMA trend filter and volume expansion (>1.5x average) to enter trades. Designed for low trade frequency (15-25/year) to minimize fee drift while capturing mean reversion in ranging markets and momentum in trending markets. Works in both bull and bear markets by adapting to 1-day trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_vwap_bounce_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h VWAP calculation
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # 12h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1-day EMA trend filter ===
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: bullish when EMA20 > EMA50, bearish when EMA20 < EMA50
    trend_bullish = ema_20_1d > ema_50_1d
    trend_bearish = ema_20_1d < ema_50_1d
    
    # Align 1-day trends to 12h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish)
    
    # Session filter: 08-20 UTC (major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(vwap[i]) or np.isnan(atr_12h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_vwap = vwap[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 12h volume must be elevated (1.5x 20-period average)
        volume_elevated = volume_current > 1.5 * vol_ma
        
        # Price relative to VWAP
        price_above_vwap = price_close > price_vwap
        price_below_vwap = price_close < price_vwap
        
        # Long conditions: price crosses above VWAP with volume expansion + 1-day bullish trend
        long_signal = volume_elevated and price_above_vwap and trend_bullish_aligned[i]
        
        # Short conditions: price crosses below VWAP with volume expansion + 1-day bearish trend
        short_signal = volume_elevated and price_below_vwap and trend_bearish_aligned[i]
        
        # Exit when price returns to VWAP (mean reversion)
        exit_long = position == 1 and price_close < price_vwap
        exit_short = position == -1 and price_close > price_vwap
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Price tends to revert to VWAP during low volatility periods when the 1-day trend is aligned. Uses 12h price crossing above/below VWAP with 1-day EMA trend filter and volume expansion (>1.5x average) to enter trades. Designed for low trade frequency (15-25/year) to minimize fee drift while capturing mean reversion in ranging markets and momentum in trending markets. Works in both bull and bear markets by adapting to 1-day trend.