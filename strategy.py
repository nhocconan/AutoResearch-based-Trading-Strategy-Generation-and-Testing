# %%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_rsi_divergence_v1
# Uses weekly RSI divergence with price to catch reversals in both bull and bear markets.
# Bullish divergence: price makes lower low, RSI makes higher low -> long signal.
# Bearish divergence: price makes higher high, RSI makes lower high -> short signal.
# Uses daily trend filter (price > EMA50 for long, price < EMA50 for short) to align with higher timeframe trend.
# Volume confirmation ensures momentum behind the move.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull markets (catching pullbacks in uptrend) and bear markets (catching rallies in downtrend).

name = "6h_1w_1d_rsi_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly RSI (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(prices, np.nan, dtype=float)
        avg_loss = np.full_like(prices, np.nan, dtype=float)
        
        # First average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    weekly_close = df_1w['close'].values
    weekly_rsi = calculate_rsi(weekly_close, 14)
    
    # Align weekly RSI to 6h timeframe
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_1w, weekly_rsi)
    
    # Daily EMA50 for trend filter
    daily_close = df_1d['close'].values
    ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Need at least 2 weekly points for divergence
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_rsi_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Get current weekly index (for divergence calculation)
        # We need to find the corresponding weekly bar index
        weekly_idx = np.sum(df_1w.index.values <= prices.index[i]) - 1
        if weekly_idx < 1:
            # Hold current position if not enough weekly data
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Get weekly values for divergence
        curr_weekly_price = weekly_close[weekly_idx]
        curr_weekly_rsi = weekly_rsi[weekly_idx]
        prev_weekly_price = weekly_close[weekly_idx-1]
        prev_weekly_rsi = weekly_rsi[weekly_idx-1]
        
        # Check for bullish divergence: price lower low, RSI higher low
        bullish_div = (curr_weekly_price < prev_weekly_price) and (curr_weekly_rsi > prev_weekly_rsi)
        # Check for bearish divergence: price higher high, RSI lower high
        bearish_div = (curr_weekly_price > prev_weekly_price) and (curr_weekly_rsi < prev_weekly_rsi)
        
        # Long conditions: bullish divergence + price above daily EMA50 + volume
        if bullish_div and close[i] > ema50_aligned[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short conditions: bearish divergence + price below daily EMA50 + volume
        elif bearish_div and close[i] < ema50_aligned[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite divergence or trend failure
        elif (bearish_div and position == 1) or (bullish_div and position == -1):
            position = 0
            signals[i] = 0.0
        elif position == 1 and close[i] < ema50_aligned[i]:
            # Exit long if price breaks below daily EMA50
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema50_aligned[i]:
            # Exit short if price breaks above daily EMA50
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
# %%