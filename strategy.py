#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to detect trend strength.
# ADX > 25 indicates trending market; < 20 indicates ranging.
# In trending markets (ADX > 25): go long when bull power > 0 and rising, short when bear power < 0 and falling.
# In ranging markets (ADX < 20): fade extreme Elder Ray values (mean reversion).
# Weekly trend filter: only trade in direction of weekly EMA(50) to avoid counter-trend moves.
# Designed to work in bull (trend following) and bear (mean reversion in ranges) markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Elder Ray: 13-period EMA of close
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    bull_power = high - ema13  # bull power: high - EMA
    bear_power = low - ema13   # bear power: low - EMA
    
    # 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).values
    dm_plus14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).values
    dm_minus14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).values
    
    # DI and DX
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Weekly trend filter: EMA(50) on weekly
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Signals
    signals = np.zeros(n)
    
    for i in range(14, n):  # Start after warmup for ADX
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Determine market regime
        is_trending = adx_14_aligned[i] > 25
        is_ranging = adx_14_aligned[i] < 20
        
        # Determine weekly trend bias
        weekly_bullish = close[i] > ema50_1w_aligned[i]
        weekly_bearish = close[i] < ema50_1w_aligned[i]
        
        # Initialize signal as hold
        signals[i] = signals[i-1] if i > 0 else 0.0
        
        if is_trending:
            # Trending market: trend following with Elder Ray
            # Long: bull power positive and rising (bulls in control)
            if bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and weekly_bullish:
                signals[i] = 0.25
            # Short: bear power negative and falling (bears in control)
            elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and weekly_bearish:
                signals[i] = -0.25
        elif is_ranging:
            # Ranging market: mean reversion at extreme Elder Ray
            # Calculate z-score of bull/bear power over last 50 periods
            bull_ma = pd.Series(bull_power).rolling(window=50, min_periods=20).mean()[i]
            bull_std = pd.Series(bull_power).rolling(window=50, min_periods=20).std()[i]
            bear_ma = pd.Series(bear_power).rolling(window=50, min_periods=20).mean()[i]
            bear_std = pd.Series(bear_power).rolling(window=50, min_periods=20).std()[i]
            
            if not (np.isnan(bull_ma) or np.isnan(bull_std) or 
                    np.isnan(bear_ma) or np.isnan(bear_std)):
                bull_z = (bull_power[i] - bull_ma) / bull_std if bull_std > 0 else 0
                bear_z = (bear_power[i] - bear_ma) / bear_std if bear_std > 0 else 0
                
                # Long: extreme bear power (oversold) in ranging market
                if bear_z < -1.5 and weekly_bullish:  # not strongly against weekly trend
                    signals[i] = 0.25
                # Short: extreme bull power (overbought) in ranging market
                elif bull_z > 1.5 and weekly_bearish:  # not strongly against weekly trend
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_WeeklyFilter"
timeframe = "6h"
leverage = 1.0