#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily KAMA + RSI + Chop Filter
# Hypothesis: KAMA identifies trend direction with low whipsaw, RSI filters overbought/oversold,
# and Choppiness Index avoids choppy markets. Works in both bull (trend following) and bear
# (mean reversion in ranges) by adapting to market regime.
# Target: 15-25 trades/year (60-100 over 4 years).

name = "daily_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.abs(np.diff(close, n=1))  # |close[t] - close[t-1]|
    
    # Pad arrays for calculation
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility])
    
    er = np.full_like(change_padded, np.nan, dtype=float)
    for i in range(10, len(change_padded)):
        if not np.isnan(change_padded[i]) and not np.isnan(volatility_padded[i-9:i+1]).any():
            sum_vol = np.nansum(volatility_padded[i-9:i+1])
            if sum_vol > 0:
                er[i] = change_padded[i] / sum_vol
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9
    for i in range(10, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Shift KAMA by 1 to avoid look-ahead
    kama = np.roll(kama, 1)
    if len(kama) > 1:
        kama[0] = kama[1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # First average (simple mean)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
    
    # Wilder smoothing
    for i in range(14, len(close)):
        if not np.isnan(avg_gain[i-1]) and not np.isnan(avg_loss[i-1]):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.roll(rsi, 1)
    if len(rsi) > 1:
        rsi[0] = rsi[1]
    
    # Calculate Choppiness Index (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # ATR(14)
    atr = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Sum of ATR over 14 periods
    atr_sum = np.full_like(close, np.nan)
    for i in range(27, len(close)):  # Start from index 27 (14+13)
        if not np.isnan(atr[i-13:i+1]).any():
            atr_sum[i] = np.nansum(atr[i-13:i+1])
    
    # Max-min range over 14 periods
    max_high = np.full_like(close, np.nan)
    min_low = np.full_like(close, np.nan)
    for i in range(13, len(close)):
        max_high[i] = np.nanmax(high[i-13:i+1])
        min_low[i] = np.nanmin(low[i-13:i+1])
    
    # Choppiness Index
    chop = np.full_like(close, 50.0)  # Default to neutral
    for i in range(27, len(close)):
        if not np.isnan(atr_sum[i]) and not np.isnan(max_high[i]) and not np.isnan(min_low[i]):
            range_val = max_high[i] - min_low[i]
            if range_val > 0 and atr_sum[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / range_val) / np.log10(14)
    
    # Shift indicators by 1 to avoid look-ahead
    chop = np.roll(chop, 1)
    if len(chop) > 1:
        chop[0] = chop[1]
    
    # Align weekly trend filter (optional: use weekly close vs EMA for regime)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) >= 50:
        weekly_close = df_weekly['close'].values
        weekly_close_series = pd.Series(weekly_close)
        weekly_ema_50 = weekly_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
        weekly_ema_50 = np.roll(weekly_ema_50, 1)
        if len(weekly_ema_50) > 1:
            weekly_ema_50[0] = weekly_ema_50[1]
        weekly_ema_50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_50)
    else:
        weekly_ema_50_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            (len(df_weekly) >= 50 and np.isnan(weekly_ema_50_aligned[i]))):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chop < 38.2 = trending, Chop > 61.8 = ranging
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit conditions
            exit_condition = False
            if is_trending:
                # In trending market: exit if price crosses below KAMA
                if close[i] < kama[i]:
                    exit_condition = True
            else:
                # In ranging market: exit if RSI > 70 (overbought)
                if rsi[i] > 70:
                    exit_condition = True
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_condition = False
            if is_trending:
                # In trending market: exit if price crosses above KAMA
                if close[i] > kama[i]:
                    exit_condition = True
            else:
                # In ranging market: exit if RSI < 30 (oversold)
                if rsi[i] < 30:
                    exit_condition = True
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
                
        else:  # Flat, look for entry
            # Determine market regime
            if is_trending:
                # Trend following mode
                # Long: price > KAMA and RSI > 50 (bullish momentum)
                # Short: price < KAMA and RSI < 50 (bearish momentum)
                if close[i] > kama[i] and rsi[i] > 50:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < kama[i] and rsi[i] < 50:
                    position = -1
                    signals[i] = -0.25
            else:
                # Mean reversion mode (ranging market)
                # Long: RSI < 30 (oversold)
                # Short: RSI > 70 (overbought)
                if rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals