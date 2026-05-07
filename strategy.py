#!/usr/bin/env python3
name = "1h_RSI_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Load 1d data ONCE for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d volume SMA20 for volume filter
    vol_sma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # 1h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: RSI < 30 (oversold) + 4h uptrend + volume above average
            if (rsi[i] < 30 and 
                ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] and 
                volume[i] > vol_sma_20_1d_aligned[i] and
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + 4h downtrend + volume above average
            elif (rsi[i] > 70 and 
                  ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] and 
                  volume[i] > vol_sma_20_1d_aligned[i] and
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: RSI > 50 or trend change
            if rsi[i] > 50 or ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI < 50 or trend change
            if rsi[i] < 50 or ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h RSI mean reversion with 4h trend filter and 1d volume confirmation
# - RSI < 30 for long, RSI > 70 for short (mean reversion)
# - 4h EMA34 trend filter ensures trades align with higher timeframe momentum
# - 1d volume spike filter increases signal reliability
# - Session filter (08-20 UTC) reduces noise during low liquidity hours
# - Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend)
# - Position size 0.20 limits drawdown while maintaining sufficient exposure
# - Target: 60-120 total trades over 4 years (15-30/year) to avoid fee drag
# - Simple, robust logic with clear entry/exit conditions
# - Avoids overtrading by requiring multiple confluence factors
# - Uses higher timeframes for direction, lower timeframe only for timing precision