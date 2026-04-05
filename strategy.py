#!/usr/bin/env python3
"""
Experiment #9387: 6h Donchian(20) breakout + 1d/1w trend filter + volume confirmation.
Hypothesis: Donchian breakouts capture trends; higher timeframe filters (1d/1w) ensure directional alignment across market regimes; volume confirms institutional participation. Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost. Works in bull (breakouts with trend) and bear (filtered shorts).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_9387_6h_donchian20_1d_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_PERIOD_DAILY = 50
TREND_PERIOD_WEEKLY = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_PERIOD_DAILY, adjust=False, min_periods=TREND_PERIOD_DAILY).mean().values
    
    # Calculate 1w EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=TREND_PERIOD_WEEKLY, adjust=False, min_periods=TREND_PERIOD_WEEKLY).mean().values
    
    # Price relative to EMAs: above = bullish bias, below = bearish bias
    price_vs_ema_1d = np.where(close_1d > ema_1d, 1, 
                       np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_1w = np.where(close_1w > ema_1w, 1, 
                       np.where(close_1w < ema_1w, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    
    price_vs_ema_1d_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema_1d)
    price_vs_ema_1w_aligned = align_htf_to_ltf(prices, df_1w, price_vs_ema_1w)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_PERIOD_DAILY, TREND_PERIOD_WEEKLY, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_1d_aligned[i]) or np.isnan(price_vs_ema_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d and 1w EMA (both must agree)
        bull_bias = (price_vs_ema_1d_aligned[i] == 1) and (price_vs_ema_1w_aligned[i] == 1)
        bear_bias = (price_vs_ema_1d_aligned[i] == -1) and (price_vs_ema_1w_aligned[i] == -1)
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals