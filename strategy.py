#!/usr/bin/env python3
# mtf_1h_ema_rsi_pullback_4h1d_v1
# Hypothesis: 1h strategy using 4h EMA(21) for trend direction and daily RSI(14) for mean-reversion entries. Enters long on pullbacks to 4h EMA when daily RSI < 30 (oversold), enters short on rallies to 4h EMA when daily RSI > 70 (overbought). Uses session filter (08-20 UTC) to avoid low-liquidity hours. Position size fixed at 0.20 to limit fee churn. Target: 60-150 total trades over 4 years (15-37/year). Works in bull markets via trend-following pullsbacks and in bear markets via mean-reversion bounces off HTF trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_ema_rsi_pullback_4h1d_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Multi-timeframe: 4h EMA(21) trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_4h_s = pd.Series(close_4h)
    ema_21_4h = close_4h_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Multi-timeframe: daily RSI(14)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    delta = close_1d_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_values = rsi_14_1d.values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 4h EMA or RSI > 70 (overbought)
            if close[i] < ema_21_4h_aligned[i] or rsi_14_1d_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 4h EMA or RSI < 30 (oversold)
            if close[i] > ema_21_4h_aligned[i] or rsi_14_1d_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for pullback to 4h EMA with RSI extreme
            near_ema = abs(close[i] - ema_21_4h_aligned[i]) / ema_21_4h_aligned[i] < 0.005  # Within 0.5%
            oversold = rsi_14_1d_aligned[i] < 30
            overbought = rsi_14_1d_aligned[i] > 70
            
            if near_ema and oversold:
                position = 1
                signals[i] = 0.20
            elif near_ema and overbought:
                position = -1
                signals[i] = -0.20
    
    return signals