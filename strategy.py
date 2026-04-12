# 1h_4h_1d_cam_rsi_hybrid_v1
# Hybrid: 4h trend (EMA21), 1d RSI for mean reversion in range, 1h Candlestick for entry timing
# Target: 15-37 trades/year (60-150 total) with 0.20 position sizing
# Rationale: Combines trend-following and mean-reversion to work in both bull/bear markets.
# Uses 1d RSI <30/ >70 for reversals, 4h EMA21 for trend filter, 1h hammer/shooting star for entry.
# Session filter 08-20 UTC to avoid low-liquidity hours.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_cam_rsi_hybrid_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA21 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1d data for RSI(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # Calculate RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if np.isnan(ema21_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Candlestick patterns for 1h entry
        body = abs(close[i] - open_price[i])
        lower_wick = min(open_price[i], close[i]) - low[i]
        upper_wick = high[i] - max(open_price[i], close[i])
        is_hammer = (lower_wick > 2 * body) and (upper_wick < 0.1 * body)
        is_shooting_star = (upper_wick > 2 * body) and (lower_wick < 0.1 * body)
        
        # Long conditions: oversold RSI + hammer + above 4h EMA21
        long_signal = (rsi_1d_aligned[i] < 30) and is_hammer and (close[i] > ema21_4h_aligned[i])
        # Short conditions: overbought RSI + shooting star + below 4h EMA21
        short_signal = (rsi_1d_aligned[i] > 70) and is_shooting_star and (close[i] < ema21_4h_aligned[i])
        
        # Exit conditions: RSI reverts to 50 or opposite candle
        exit_long = (rsi_1d_aligned[i] > 50) or is_shooting_star
        exit_short = (rsi_1d_aligned[i] < 50) or is_hammer
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals