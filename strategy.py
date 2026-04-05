#!/usr/bin/env python3
"""
Experiment #7674: 1-hour RSI mean reversion with 4h trend filter and 1d volatility filter.
Hypothesis: In trending markets (price > 4h EMA50), buy RSI pullbacks below 30.
In ranging markets (price near 4h EMA50), sell RSI extremes (>70 or <30).
Volatility filter: only trade when 1d ATR ratio > 0.8 (avoid low volatility chop).
Timeframe: 1h. Target: 80-150 trades over 4 years (20-38/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7674_1h_rsi_meanrev_4h_ema50_1d_atr_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_TREND = 50
ATR_PERIOD = 14
VOLATILITY_LOOKBACK = 30
VOLATILITY_THRESHOLD = 0.8
SIGNAL_SIZE = 0.20

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=VOLATILITY_LOOKBACK, min_periods=VOLATILITY_LOOKBACK).mean().values
    volatility_ratio = atr_1d / atr_ma_1d
    volatility_ratio_aligned = align_htf_to_ltf(prices, df_1d, volatility_ratio)
    
    # Calculate LTF indicators
    close = prices['close'].values
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(RSI_PERIOD, EMA_TREND, VOLATILITY_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_50_aligned[i]) or np.isnan(volatility_ratio_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is above threshold
        volatility_filter = volatility_ratio_aligned[i] > VOLATILITY_THRESHOLD
        
        # Trend determination
        price_vs_ema = close[i] - ema_4h_50_aligned[i]
        trend_strength = abs(price_vs_ema) / ema_4h_50_aligned[i]
        is_trending = trend_strength > 0.02  # 2% deviation from EMA
        is_ranging = trend_strength <= 0.02
        
        # RSI conditions
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Entry logic
        if position == 0:
            if volatility_filter:
                if is_trending and rsi_oversold:
                    # In uptrend, buy oversold
                    signals[i] = SIGNAL_SIZE
                    position = 1
                elif is_trending and rsi_overbought:
                    # In downtrend, sell overbought
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                elif is_ranging:
                    # In ranging, fade extremes
                    if rsi_oversold:
                        signals[i] = SIGNAL_SIZE
                        position = 1
                    elif rsi_overbought:
                        signals[i] = -SIGNAL_SIZE
                        position = -1
        elif position == 1:
            # Exit long when RSI returns to neutral
            if rsi[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when RSI returns to neutral
            if rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals