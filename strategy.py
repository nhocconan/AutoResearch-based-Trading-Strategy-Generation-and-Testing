#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_RegimeFilter
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volatility regime filter.
In trending markets (price > 4h EMA50 + low volatility), trade breakouts in direction of trend.
In ranging markets (high volatility or weak trend), fade at Camarilla extremes.
Uses discrete sizing (0.20) to minimize fee churn. Target: 15-30 trades/year per symbol.
Works in bull via trend-following breakouts, in bear via mean reversion at extremes when trend weakens.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculations and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar (based on previous bar)
    R1_4h = np.full(len(close_4h), np.nan)
    S1_4h = np.full(len(close_4h), np.nan)
    R2_4h = np.full(len(close_4h), np.nan)
    S2_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        # Camarilla levels based on previous 4h bar's range
        high_prev = high_4h[i-1]
        low_prev = low_4h[i-1]
        close_prev = close_4h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R2_4h[i] = close_prev + (range_prev * 1.1 / 2)  # R2 level
            S2_4h[i] = close_prev - (range_prev * 1.1 / 2)  # S2 level
            R1_4h[i] = close_prev + (range_prev * 1.1 / 4)  # R1 level
            S1_4h[i] = close_prev - (range_prev * 1.1 / 4)  # S1 level
    
    # Align Camarilla levels to original timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    R2_4h_aligned = align_htf_to_ltf(prices, df_4h, R2_4h)
    S2_4h_aligned = align_htf_to_ltf(prices, df_4h, S2_4h)
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volatility regime filter: ATR ratio (current ATR vs 24-period average)
    # High ATR = choppy/ranging market, Low ATR = trending market
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First bar
    atr_current = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_24 = pd.Series(tr).rolling(window=24, min_periods=24).mean().values
    atr_ratio = atr_current / atr_ma_24
    # Low volatility regime: ATR ratio < 0.8 (trending)
    # High volatility regime: ATR ratio > 1.2 (choppy/ranging)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_ratio[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_trend = ema_50_4h_aligned[i]
        volatility_regime = atr_ratio[i]  # < 0.8 = trending, > 1.2 = ranging
        
        if position == 0:
            # Regime-based entry logic
            if volatility_regime < 0.8:  # Low volatility = trending market
                # Trend-following: break in direction of 4h EMA50 trend
                if close[i] > ema_trend:  # Uptrend
                    long_signal = close[i] > R1_4h_aligned[i]
                    short_signal = False  # No counter-trend in strong trend
                else:  # Downtrend
                    short_signal = close[i] < S1_4h_aligned[i]
                    long_signal = False  # No counter-trend in strong trend
            else:  # High volatility = ranging/choppy market
                # Mean reversion: fade at Camarilla extremes
                long_signal = close[i] < S1_4h_aligned[i]  # Buy at support
                short_signal = close[i] > R1_4h_aligned[i]  # Sell at resistance
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions: touch opposite level or trend reversal
            if volatility_regime < 0.8:  # Trending market
                exit_signal = close[i] < ema_trend * 0.995  # Trend reversal
            else:  # Ranging market
                exit_signal = close[i] > (R1_4h_aligned[i] + S1_4h_aligned[i]) / 2  # Mean reversion to midpoint
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions: touch opposite level or trend reversal
            if volatility_regime < 0.8:  # Trending market
                exit_signal = close[i] > ema_trend * 1.005  # Trend reversal
            else:  # Ranging market
                exit_signal = close[i] < (R1_4h_aligned[i] + S1_4h_aligned[i]) / 2  # Mean reversion to midpoint
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_RegimeFilter"
timeframe = "1h"
leverage = 1.0