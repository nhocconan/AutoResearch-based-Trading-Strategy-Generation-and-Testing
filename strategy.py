#!/usr/bin/env python3
name = "6h_MultiTimeframe_Momentum_Confluence_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly trend: 50-period EMA
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily momentum: RSI(14)
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 6h momentum: Price rate of change over 4 periods (24 hours)
    roc_4 = np.zeros_like(close)
    for i in range(4, n):
        roc_4[i] = (close[i] - close[i-4]) / close[i-4] * 100
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # ~3 days for 6h to reduce trades
    
    start_idx = max(50, 20, 4)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine multi-timeframe conditions
        weekly_uptrend = close > ema_50_1w_aligned[i]
        weekly_downtrend = close < ema_50_1w_aligned[i]
        daily_momentum_strong = rsi_14_1d_aligned[i] > 55  # Bullish momentum
        daily_momentum_weak = rsi_14_1d_aligned[i] < 45   # Bearish momentum
        price_momentum_pos = roc_4[i] > 0.5   # Positive short-term momentum
        price_momentum_neg = roc_4[i] < -0.5  # Negative short-term momentum
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Weekly uptrend + daily bullish momentum + positive price momentum + volume
            if (weekly_uptrend and 
                daily_momentum_strong and 
                price_momentum_pos and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Weekly downtrend + daily bearish momentum + negative price momentum + volume
            elif (weekly_downtrend and 
                  daily_momentum_weak and 
                  price_momentum_neg and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Weekly trend turns down OR daily momentum turns bearish
            if (not weekly_uptrend or not daily_momentum_strong):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Weekly trend turns up OR daily momentum turns bullish
            if (not weekly_downtrend or not daily_momentum_weak):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Multi-timeframe momentum confluence on 6h timeframe captures sustained trends
# while avoiding whipsaws. Weekly EMA50 establishes the primary trend direction, daily RSI
# confirms intermediate-term momentum strength, and 6-hour ROC captures short-term momentum
# alignment. Volume filter ensures institutional participation. This confluence approach
# should work in both bull (weekly uptrend + bullish momentum) and bear (weekly downtrend
# + bearish momentum) markets by requiring alignment across timeframes. Target: 15-25
# trades per year to minimize fee drag while maintaining edge. Position size 0.25 manages
# drawdown during volatile periods. The strategy avoids overtrading through strict
# multi-timeframe confirmation and cooldown periods.