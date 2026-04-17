#!/usr/bin/env python3
"""
1d_Position_Size_Management_v1
Daily strategy with position sizing based on volatility regime (ATR-based) and trend strength (ADX).
Uses 1-week trend filter (EMA34) to avoid counter-trend trades.
Positions scaled: 0.15 in low volatility/weak trend, 0.30 in high volatility/strong trend.
Designed for low trade frequency (<25/year) and resilience in bull/bear markets.
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
    
    # === 1-week Trend Filter (EMA34) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate EMA34 on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily ATR (14) for volatility regime ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Daily ADX (14) for trend strength ===
    # +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # First element needs handling
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX (already daily, but ensure alignment for consistency)
    adx_aligned = adx  # Already aligned to daily prices
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(adx[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine trend direction from weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volatility regime: normalize ATR by price
        atr_ratio = atr[i] / close[i] if close[i] > 0 else 0
        high_vol = atr_ratio > 0.02  # >2% daily ATR
        
        # Trend strength regime
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # Position sizing logic
        target_size = 0.0
        if uptrend and strong_trend and high_vol:
            # Strong uptrend with high volatility -> full position
            target_size = 0.30
        elif uptrend and (weak_trend or not high_vol):
            # Weak trend or low volatility -> half position
            target_size = 0.15
        elif downtrend and strong_trend and high_vol:
            # Strong downtrend with high volatility -> full short
            target_size = -0.30
        elif downtrend and (weak_trend or not high_vol):
            # Weak trend or low volatility -> half short
            target_size = -0.15
        # In ranging markets (weak trend) -> stay flat or small position
        elif weak_trend:
            target_size = 0.0
        
        # Only change position if signal differs significantly to reduce churn
        if position == 0:
            if target_size != 0:
                signals[i] = target_size
                position = 1 if target_size > 0 else -1
        elif position == 1:
            if target_size <= 0:
                signals[i] = 0.0
                position = 0
            elif abs(target_size - 0.30) > 0.05:  # Significant change
                signals[i] = target_size
                position = 1 if target_size > 0 else -1
            else:
                signals[i] = 0.30  # Maintain long
        elif position == -1:
            if target_size >= 0:
                signals[i] = 0.0
                position = 0
            elif abs(target_size + 0.30) > 0.05:  # Significant change
                signals[i] = target_size
                position = 1 if target_size > 0 else -1
            else:
                signals[i] = -0.30  # Maintain short
    
    return signals

name = "1d_Position_Size_Management_v1"
timeframe = "1d"
leverage = 1.0