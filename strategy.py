#!/usr/bin/env python3
"""
1h Volume-Weighted RSI with 4h EMA Trend and 1d Volatility Filter
Combines VW-RSI (7) for mean-reversion entries, 4h EMA50 for trend alignment,
and 1d ATR percentile to filter low-volatility chop. Designed for 15-30 trades/year
with discrete sizing to minimize fee drag. Works in both bull (dips in uptrend)
and bear (bounces in downtrend) by fading extremes only when trend agrees.
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
    
    # Calculate VW-RSI(7): RSI on typical price weighted by volume
    typical_price = (high + low + close) / 3.0
    change = np.diff(typical_price, prepend=typical_price[0])
    pos_change = np.where(change > 0, change, 0.0)
    neg_change = np.where(change < 0, -change, 0.0)
    
    # Volume-weighted smoothing
    vol_pos = pos_change * volume
    vol_neg = neg_change * volume
    
    vol_pos_ema = pd.Series(vol_pos).ewm(alpha=1/7, adjust=False).values
    vol_neg_ema = pd.Series(vol_neg).ewm(alpha=1/7, adjust=False).values
    
    rs = vol_pos_ema / (vol_neg_ema + 1e-10)
    vwrsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for ATR percentile volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.absolute(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align with index
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR percentile rank (50-day lookback) - avoid low volatility
    atr_percentile = pd.Series(atr_14).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwrsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = vwrsi[i]
        ema_trend = ema_50_4h_aligned[i]
        vol_filter = atr_percentile_aligned[i]
        
        # Only trade when volatility is sufficient (above 30th percentile)
        if vol_filter < 0.3:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: VW-RSI oversold (<30) and price above 4h EMA (uptrend)
            if rsi_val < 30.0 and price > ema_trend:
                signals[i] = 0.20
                position = 1
            # Short: VW-RSI overbought (>70) and price below 4h EMA (downtrend)
            elif rsi_val > 70.0 and price < ema_trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position: hold until RSI reaches 50 (mean reversion target)
            signals[i] = 0.20
            if rsi_val >= 50.0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until RSI reaches 50
            signals[i] = -0.20
            if rsi_val <= 50.0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_VWRSI_4hEMA50_1dATRPercentile_Filter"
timeframe = "1h"
leverage = 1.0