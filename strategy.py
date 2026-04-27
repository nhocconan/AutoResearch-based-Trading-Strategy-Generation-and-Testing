#!/usr/bin/env python3
"""
Strategy: 1d_Fibonacci_Retracement_1wTrend
Timeframe: 1d
Hypothesis: 
- In trending markets, price often retraces to key Fibonacci levels (38.2%, 50%, 61.8%) before continuing the trend.
- Use 1-week trend as filter to avoid counter-trend trades.
- Enter long at 61.8% retracement of weekly uptrend, short at 38.2% retracement of weekly downtrend.
- Add volume confirmation to avoid false breakouts.
- Target: 10-25 trades/year to minimize fee drag.
- Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for weekly lookback
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Fibonacci levels (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA 21 for trend direction
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly RSI for momentum confirmation
    delta_1w = pd.Series(close_1w).diff()
    gain_1w = delta_1w.where(delta_1w > 0, 0)
    loss_1w = -delta_1w.where(delta_1w < 0, 0)
    avg_gain_1w = gain_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1w = loss_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1w = avg_gain_1w / avg_loss_1w
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate weekly ATR for volatility filter
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                       np.abs(high_1w[1:] - close_1w[:-1]), 
                       np.abs(low_1w[1:] - close_1w[:-1]))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly volume moving average
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Calculate weekly Fibonacci retracement levels
    # For each point, we need the recent weekly swing high and low
    lookback = 26  # ~6 months of weekly data
    fib_levels_long = np.full(n, np.nan)  # 61.8% level for longs (in uptrend)
    fib_levels_short = np.full(n, np.nan)  # 38.2% level for shorts (in downtrend)
    
    for i in range(n):
        # Get corresponding weekly index (approximate)
        weekly_idx = min(i // 7, len(close_1w) - 1)  # ~7 daily bars per week
        
        if weekly_idx < lookback:
            continue
            
        # Get lookback window for swing high/low
        start_idx = max(0, weekly_idx - lookback)
        end_idx = weekly_idx
        
        if end_idx <= start_idx:
            continue
            
        # Find swing high and low in the weekly window
        swing_high = np.max(high_1w[start_idx:end_idx+1])
        swing_low = np.min(low_1w[start_idx:end_idx+1])
        
        # Calculate Fibonacci levels
        diff = swing_high - swing_low
        if diff <= 0:
            continue
            
        # 61.8% retracement for longs (in uptrend)
        fib_618 = swing_high - 0.618 * diff
        # 38.2% retracement for shorts (in downtrend)
        fib_382 = swing_low + 0.382 * diff
        
        # Store the levels (they apply until next weekly update)
        fib_levels_long[i] = fib_618
        fib_levels_short[i] = fib_382
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(fib_levels_long[i]) or
            np.isnan(fib_levels_short[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 21-week EMA
        price_above_ema = close[i] > ema_21_1w_aligned[i]
        price_below_ema = close[i] < ema_21_1w_aligned[i]
        
        # RSI filter: avoid extreme conditions
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30
        
        # Volatility filter: only trade when volatility is reasonable
        vol_filter = atr_1w_aligned[i] > 0
        
        # Volume filter: current volume above weekly average
        volume_filter = volume[i] > vol_ma_1w_aligned[i]
        
        # Fibonacci level proximity (within 0.5% of level)
        fib_long_level = fib_levels_long[i]
        fib_short_level = fib_levels_short[i]
        
        near_fib_long = abs(close[i] - fib_long_level) / close[i] < 0.005 if not np.isnan(fib_long_level) else False
        near_fib_short = abs(close[i] - fib_short_level) / close[i] < 0.005 if not np.isnan(fib_short_level) else False
        
        # Long conditions: 
        # - Uptrend (price above weekly EMA21)
        # - Near 61.8% Fibonacci retracement level
        # - RSI not overbought
        # - Volume confirmation
        long_condition = (price_above_ema and 
                         near_fib_long and
                         rsi_not_overbought and
                         volume_filter and
                         vol_filter)
        
        # Short conditions:
        # - Downtrend (price below weekly EMA21)
        # - Near 38.2% Fibonacci retracement level
        # - RSI not oversold
        # - Volume confirmation
        short_condition = (price_below_ema and
                          near_fib_short and
                          rsi_not_oversold and
                          volume_filter and
                          vol_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_ema:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_ema:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Fibonacci_Retracement_1wTrend"
timeframe = "1d"
leverage = 1.0