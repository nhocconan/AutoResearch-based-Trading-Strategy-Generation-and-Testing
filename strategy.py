#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_cci_extreme_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly OHLC for weekly trend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily OHLC for daily CCI
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 6h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend: price above/below 50-week SMA
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Daily CCI(20) - uses typical price and mean deviation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    sma_tp_20 = pd.Series(typical_price_1d).rolling(window=20, min_periods=20).mean().values
    mean_deviation = pd.Series(typical_price_1d).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_1d = (typical_price_1d - sma_tp_20) / (0.015 * mean_deviation)
    # Shift by 1 to use only completed daily bars
    cci_1d = np.roll(cci_1d, 1)
    cci_1d[0] = np.nan
    # Align daily CCI to 6h timeframe
    cci_6h = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Session filter: 0-23 UTC (6h bars cover full day)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    # Minimum holding period: 2 bars (12 hours) to reduce churn
    hold_count = np.zeros(n, dtype=int)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Decrease hold counter
        if hold_count[i] > 0:
            hold_count[i] -= 1
        
        # Skip if any required data is invalid or outside session or holding
        if (np.isnan(cci_6h[i]) or np.isnan(sma_50_1w_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i] or hold_count[i] > 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 6h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Weekly trend filter
        weekly_uptrend = price_close > sma_50_1w_aligned[i]
        weekly_downtrend = price_close < sma_50_1w_aligned[i]
        
        # Extreme CCI levels for mean reversion
        cci_overbought = cci_6h[i] > 150
        cci_oversold = cci_6h[i] < -150
        
        # Long conditions: CCI oversold with volume and weekly uptrend
        long_signal = volume_confirmed and cci_oversold and weekly_uptrend
        
        # Short conditions: CCI overbought with volume and weekly downtrend
        short_signal = volume_confirmed and cci_overbought and weekly_downtrend
        
        # Exit when CCI returns to neutral zone (-50 to 50)
        exit_long = position == 1 and cci_6h[i] > -50
        exit_short = position == -1 and cci_6h[i] < 50
        
        # Trading logic with minimum holding period
        if long_signal and position != 1:
            position = 1
            hold_count[i] = 2  # Hold for 2 bars minimum
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            hold_count[i] = 2  # Hold for 2 bars minimum
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

# Hypothesis: 6h CCI extreme reversal with weekly trend filter and volume confirmation.
# Uses daily CCI(20) to identify extreme overbought/oversold conditions (>150, <-150).
# Enters long when CCI is deeply oversold (<-150) with volume >1.5x 20-period average and price above weekly SMA50.
# Enters short when CCI is deeply overbought (>150) with volume >1.5x 20-period average and price below weekly SMA50.
# Exits when CCI returns to neutral zone (-50 to 50), capturing mean reversion within the daily range.
# Weekly SMA50 filter ensures trades align with higher timeframe trend, reducing counter-trend trades.
# Volume filter reduces false signals from low-volume extremes.
# Designed to work in both bull and bear markets by fading extremes in the direction of weekly trend.
# Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag.