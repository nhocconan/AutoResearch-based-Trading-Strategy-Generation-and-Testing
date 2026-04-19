#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 38.2% Fibonacci retracement of weekly trend + volume confirmation + RSI filter
# In bull markets, price pulls back to Fibonacci support in uptrend; in bear markets, rallies to resistance.
# Weekly trend direction determined by price vs weekly EMA50. Enter long when price >= 38.2% retracement level of weekly uptrend,
# short when price <= 61.8% retracement level of weekly downtrend. Volume > 1.5x average confirms interest.
# RSI(14) between 40-60 avoids overbought/oversold extremes. Target: 15-25 trades/year.
name = "1d_Fib382_WeeklyTrend_VolumeRSI"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Fibonacci levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Weekly EMA50 for trend direction
    ema_50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly swing high/low for Fibonacci levels (using last 20 weekly bars)
    def calculate_swing_points(high, low, lookback=20):
        swing_high = np.full_like(high, np.nan)
        swing_low = np.full_like(low, np.nan)
        for i in range(lookback, len(high)):
            swing_high[i] = np.max(high[i-lookback:i])
            swing_low[i] = np.min(low[i-lookback:i])
        return swing_high, swing_low
    
    swing_high, swing_low = calculate_swing_points(weekly_high, weekly_low, 20)
    
    # Fibonacci levels: 38.2% and 61.8% retracement
    fib_range = swing_high - swing_low
    fib_382 = swing_low + 0.382 * fib_range
    fib_618 = swing_low + 0.618 * fib_range
    
    # Align weekly data to daily
    ema_50_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    fib_382_aligned = align_htf_to_ltf(prices, df_weekly, fib_382)
    fib_618_aligned = align_htf_to_ltf(prices, df_weekly, fib_618)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI(14) to avoid extremes
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure weekly EMA50 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(fib_382_aligned[i]) or 
            np.isnan(fib_618_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_aligned[i]
        fib_382_val = fib_382_aligned[i]
        fib_618_val = fib_618_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        rsi_val = rsi[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # RSI filter: avoid overbought/oversold
        rsi_filter = (rsi_val >= 40) & (rsi_val <= 60)
        
        # Determine weekly trend: price vs EMA50
        weekly_uptrend = ema_50_val > fib_382_val  # Simplified: if EMA50 above 38.2 level, assume uptrend
        weekly_downtrend = ema_50_val < fib_618_val  # If EMA50 below 61.8 level, assume downtrend
        
        if position == 0:
            # Enter long: weekly uptrend, price at or above 38.2% retracement, volume confirmed, RSI OK
            if weekly_uptrend and price >= fib_382_val and volume_confirmed and rsi_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend, price at or below 61.8% retracement, volume confirmed, RSI OK
            elif weekly_downtrend and price <= fib_618_val and volume_confirmed and rsi_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below 38.2% level
            if not weekly_uptrend or price < fib_382_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above 61.8% level
            if not weekly_downtrend or price > fib_618_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals