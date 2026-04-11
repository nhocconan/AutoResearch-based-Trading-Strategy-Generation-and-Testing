#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Trend Pullback with 4h/1d Confluence
# Long when 4h trend is up (price > EMA21_4h), 1d momentum is positive (close > open), and 1h pulls back to EMA50
# Short when 4h trend is down (price < EMA21_4h), 1d momentum is negative (close < open), and 1h bounces to EMA50
# Uses higher timeframe for trend direction and lower timeframe for precise entry during pullbacks.
# Designed for 15-35 trades/year on 1h timeframe with strict confluence to avoid overtrading.

name = "1h_4h_1d_trend_pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 25 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1d momentum (close > open for bullish, close < open for bearish)
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    bullish_1d = close_1d > open_1d
    bearish_1d = close_1d < open_1d
    bullish_1d_aligned = align_htf_to_ltf(prices, df_4h, bullish_1d.astype(float))  # align to 4h then to 1h via 4h alignment
    bearish_1d_aligned = align_htf_to_ltf(prices, df_4h, bearish_1d.astype(float))
    
    # Calculate 1h EMA50 for pullback entries
    ema_50_1h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA50 period
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0  # close position at session end
                position = 0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(bullish_1d_aligned[i]) or 
            np.isnan(bearish_1d_aligned[i]) or np.isnan(ema_50_1h[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h trend direction
        is_uptrend_4h = close[i] > ema_21_4h_aligned[i]
        is_downtrend_4h = close[i] < ema_21_4h_aligned[i]
        
        # Get 1d momentum (using 4h alignment as proxy for 1h)
        is_bullish_1d = bullish_1d_aligned[i] > 0.5
        is_bearish_1d = bearish_1d_aligned[i] > 0.5
        
        # Entry conditions: 4h trend + 1d momentum + 1h pullback to EMA50
        # Long: 4h uptrend, 1d bullish, price pulls back to touch or cross above EMA50
        # Short: 4h downtrend, 1d bearish, price bounces to touch or cross below EMA50
        long_signal = is_uptrend_4h and is_bullish_1d and low[i] <= ema_50_1h[i] and close[i] > ema_50_1h[i]
        short_signal = is_downtrend_4h and is_bearish_1d and high[i] >= ema_50_1h[i] and close[i] < ema_50_1h[i]
        
        # Exit conditions: trend reversal or momentum divergence
        exit_long = not is_uptrend_4h or not is_bullish_1d
        exit_short = not is_downtrend_4h or not is_bearish_1d
        
        # Priority: entry > exit > hold
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals