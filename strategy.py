#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ADX trend strength + 1d RSI extremes for mean reversion in ranging markets
# Long when 1d ADX < 25 (weak trend/ranging) AND 1d RSI < 30 (oversold) AND 6h close > 6h EMA20 (short-term bias)
# Short when 1d ADX < 25 (weak trend/ranging) AND 1d RSI > 70 (overbought) AND 6h close < 6h EMA20 (short-term bias)
# Exit when 1d RSI crosses back through 50 (mean reversion to midpoint)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe
# Works in both bull (buy oversold dips in range) and bear (sell overbought rallies in range) markets
# ADX < 25 filters out strong trending markets where mean reversion fails
# RSI extremes provide high-probability reversal points in ranging/consolidation periods

name = "6h_1dADX_RSI_Extreme_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for ADX and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX/RSI
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], 100)  # Handle division by zero
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # Neutral when insufficient data
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - pd.Series(high_1d).shift(1)) > (pd.Series(low_1d).shift(1) - low_1d),
                                 np.maximum(high_1d - pd.Series(high_1d).shift(1), 0), 0))
    dm_minus = pd.Series(np.where((pd.Series(low_1d).shift(1) - low_1d) > (high_1d - pd.Series(high_1d).shift(1)),
                                  np.maximum(pd.Series(low_1d).shift(1) - low_1d, 0), 0))
    
    # Smoothed DM
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr)
    di_minus = 100 * (dm_minus_smooth / atr)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = dx.replace([np.inf, -np.inf], 0)  # Handle division by zero
    adx_1d = dx.rolling(window=14, min_periods=14).mean()
    adx_1d = adx_1d.fillna(0).values  # Zero when insufficient data
    
    # Align 1d indicators to 6h timeframe (wait for completed 1d bar)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA20 for short-term bias
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema_20_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold), ADX < 25 (ranging), price > EMA20 (short-term bull bias)
            if (rsi_1d_aligned[i] < 30 and 
                adx_1d_aligned[i] < 25 and 
                close[i] > ema_20_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought), ADX < 25 (ranging), price < EMA20 (short-term bear bias)
            elif (rsi_1d_aligned[i] > 70 and 
                  adx_1d_aligned[i] < 25 and 
                  close[i] < ema_20_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses back above 50 (mean reversion)
            if rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses back below 50 (mean reversion)
            if rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals