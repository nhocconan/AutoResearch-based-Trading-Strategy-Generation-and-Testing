#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volume regime
# - Uses 4h EMA(50) for trend direction (bull/bear filter)
# - Uses 1d ATR ratio (ATR7/ATR30) to identify volatility expansion/contraction regimes
# - In low volatility (ATR7/ATR30 < 0.8): mean revert at Bollinger Bands (20,2.0) on 1h
# - In high volatility (ATR7/ATR30 > 1.2): trend follow with 4h Donchian breakout
# - Session filter: 08-20 UTC to avoid Asian session noise
# - Discrete position sizing: ±0.20 to limit fee churn and drawdown
# - Target: 15-35 trades/year (60-140 total over 4 years) to stay within fee limits
# - Combines trend and mean reversion to work in both bull and bear markets
# - Volume regime filter prevents whipsaws in low momentum environments

name = "1h_4h_1d_regime_adaptive_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return signals
    
    # 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h Donchian(20) for breakout signals in high volatility
    highest_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Load 1d data ONCE before loop for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return signals
    
    # 1d ATR(7) and ATR(30) for volatility regime
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'], 
                       np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), 
                                  np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_7_1d = pd.Series(tr_1d).rolling(window=7, min_periods=7).mean().values
    atr_30_1d = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio_1d = atr_7_1d / atr_30_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # 1h Bollinger Bands (20,2.0) for mean reversion in low volatility
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = basis + 2.0 * dev
    lower_bb = basis - 2.0 * dev
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(highest_high_4h_aligned[i]) or 
            np.isnan(lowest_low_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(basis[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        
        # Regime determination
        atr_ratio = atr_ratio_aligned[i]
        low_vol = atr_ratio < 0.8
        high_vol = atr_ratio > 1.2
        
        # Trend filter: 4h EMA(50)
        uptrend = close_price > ema_50_4h_aligned[i]
        downtrend = close_price < ema_50_4h_aligned[i]
        
        # Initialize signals
        enter_long = False
        enter_short = False
        exit_long = False
        exit_short = False
        
        if low_vol:
            # Low volatility regime: mean reversion at Bollinger Bands
            # Long: price touches lower BB in uptrend
            if close_price <= lower_bb[i] and uptrend:
                enter_long = True
            # Short: price touches upper BB in downtrend
            if close_price >= upper_bb[i] and downtrend:
                enter_short = True
            # Exit: price returns to basis
            if position == 1 and close_price >= basis[i]:
                exit_long = True
            if position == -1 and close_price <= basis[i]:
                exit_short = True
        elif high_vol:
            # High volatility regime: trend following with Donchian breakout
            # Long: price breaks above 4h Donchian upper in uptrend
            if close_price > highest_high_4h_aligned[i] and uptrend:
                enter_long = True
            # Short: price breaks below 4h Donchian lower in downtrend
            if close_price < lowest_low_4h_aligned[i] and downtrend:
                enter_short = True
            # Exit: opposite Donchian break
            if position == 1 and close_price < lowest_low_4h_aligned[i]:
                exit_long = True
            if position == -1 and close_price > highest_high_4h_aligned[i]:
                exit_short = True
        # In neutral volatility (0.8 <= ATR ratio <= 1.2): no new entries, maintain or exit
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals