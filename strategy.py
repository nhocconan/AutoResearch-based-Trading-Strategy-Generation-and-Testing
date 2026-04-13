#!/usr/bin/env python3
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
    
    # Daily data for ATR-based volatility regime and price action
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for structural trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility regime
    atr_period = 14
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate ATR percentile rank (20-day lookback) for regime classification
    atr_rank = pd.Series(atr_1d).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Weekly higher timeframe trend: 50 EMA
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all data to 6h timeframe
    atr_rank_aligned = align_htf_to_ltf(prices, df_1d, atr_rank)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6-period RSI on 6h for mean reversion signals
    rsi_period = 6
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(atr_rank_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in low volatility environments (ATR rank < 0.3)
        low_vol_regime = atr_rank_aligned[i] < 0.3
        
        # Mean reversion conditions based on RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Trend alignment: only take mean reversion in direction of weekly trend
        price_above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long setup: oversold + low volatility + price above weekly EMA (bullish alignment)
            if rsi_oversold and low_vol_regime and price_above_weekly_ema:
                position = 1
                signals[i] = position_size
            # Short setup: overbought + low volatility + price below weekly EMA (bearish alignment)
            elif rsi_overbought and low_vol_regime and price_below_weekly_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or breaks below weekly EMA
            if rsi[i] >= 50 or close[i] <= ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral or breaks above weekly EMA
            if rsi[i] <= 50 or close[i] >= ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d1w_RSI_MeanReversion_VolatilityRegime_v1"
timeframe = "6h"
leverage = 1.0