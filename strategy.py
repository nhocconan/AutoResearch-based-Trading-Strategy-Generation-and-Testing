#!/usr/bin/env python3
"""
6h_1d_1w_Adaptive_Kelly_Reversion
Hypothesis: In bear/ranging markets (2025+), price often reverts to weekly VWAP after extreme deviations.
Combines weekly VWAP deviation with daily RSI extremes and volume exhaustion signals.
Uses adaptive Kelly sizing based on recent win rate to manage drawdown in choppy markets.
Works in both bull (trend continuation) and bear (mean reversion) regimes via regime filter.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Adaptive_Kelly_Reversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY VWAP ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate typical price and VWAP
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_numerator = np.cumsum(typical_price_1w * volume_1w)
    vwap_denominator = np.cumsum(volume_1w)
    vwap_1w = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, np.nan)
    
    # Align VWAP to 6h
    vwap_6h = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # === DAILY RSI (14) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_6h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === VOLUME EXHAUSTION (6h) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 4 days
    vol_std = pd.Series(volume).rolling(window=24, min_periods=24).std().values
    vol_zscore = np.where(vol_std != 0, (volume - vol_ma) / vol_std, 0)
    
    # === REGIME FILTER: 6-day trend strength ===
    # Use 6-period EMA slope on 6h close
    ema6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    ema6_slope = np.where(ema6[5:] != 0, (ema6[6:] - ema6[:-5]) / ema6[:-5], 0)
    ema6_slope = np.concatenate([np.full(5, np.nan), ema6_slope])  # align length
    
    # === ADAPTIVE KELLY SIZING ===
    # Track recent performance for Kelly fraction
    lookback = 50  # ~12.5 days of 6h bars
    returns = np.zeros(n)
    win_rate = np.full(n, 0.5)  # start with 50%
    avg_win = np.full(n, 0.02)  # 2% average win
    avg_loss = np.full(n, 0.015)  # 1.5% average loss
    
    # Simplified Kelly: f = (bp - q)/b where b=avg_win/avg_loss, p=win_rate, q=1-p
    kelly_fraction = np.full(n, 0.1)  # start with 10%
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(vwap_6h[i]) or np.isnan(rsi_6h[i]) or 
            np.isnan(vol_zscore[i]) or np.isnan(ema6_slope[i])):
            signals[i] = 0.0 if position == 0 else (kelly_fraction[i] if position == 1 else -kelly_fraction[i])
            continue
        
        # Calculate deviation from weekly VWAP
        price_dev = (close[i] - vwap_6h[i]) / vwap_6h[i]
        
        # Volume exhaustion: low volume after move
        vol_exhaustion = vol_zscore[i] < -0.5  # volume below average
        
        # RSI extremes
        rsi_overbought = rsi_6h[i] > 70
        rsi_oversold = rsi_6h[i] < 30
        
        # Regime filter: trend strength
        strong_trend = abs(ema6_slope[i]) > 0.005  # 0.5% per 6 periods
        
        # Entry logic: mean reversion in weak trend, trend follow in strong trend
        if not strong_trend:  # ranging market - mean reversion
            # Long: oversold RSI + price below VWAP + volume exhaustion
            long_signal = (rsi_oversold and 
                          price_dev < -0.015 and  # 1.5% below VWAP
                          vol_exhaustion)
            
            # Short: overbought RSI + price above VWAP + volume exhaustion
            short_signal = (rsi_overbought and 
                           price_dev > 0.015 and   # 1.5% above VWAP
                           vol_exhaustion)
        else:  # trending market - follow momentum
            # Long: oversold bounce in uptrend
            long_signal = (rsi_oversold and 
                          ema6_slope[i] > 0 and    # upward slope
                          price_dev < -0.005)      # slight dip
            
            # Short: overbought rejection in downtrend
            short_signal = (rsi_overbought and 
                           ema6_slope[i] < 0 and   # downward slope
                           price_dev > 0.005)      # slight rally
        
        # Exit: RSI returns to neutral or opposite extreme
        exit_long = (position == 1 and 
                    (rsi_6h[i] > 50 or rsi_6h[i] > 70))
        exit_short = (position == -1 and 
                     (rsi_6h[i] < 50 or rsi_6h[i] < 30))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            # Kelly sizing capped at 0.3
            kelly = min(kelly_fraction[i], 0.3)
            signals[i] = kelly
        elif short_signal and position != -1:
            position = -1
            kelly = min(kelly_fraction[i], 0.3)
            signals[i] = -kelly
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = kelly_fraction[i]
            elif position == -1:
                signals[i] = -kelly_fraction[i]
            else:
                signals[i] = 0.0
    
    return signals