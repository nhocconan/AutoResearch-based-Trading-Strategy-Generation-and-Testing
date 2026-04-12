#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_rsi_divergence_v1
# Uses daily RSI divergence with 4h price action for entries.
# Bullish divergence: price makes lower low, RSI makes higher low → long
# Bearish divergence: price makes higher high, RSI makes lower high → short
# Volume confirmation and ADX > 25 filter to ensure momentum.
# Designed to work in both bull (catching bounces) and bear (fading rallies) markets.
# Target: 20-40 trades/year per symbol.

name = "4h_1d_rsi_divergence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smooth(gain, 14)
    avg_loss = wilders_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # ADX trend filter: only trade when ADX > 25 (strong trend)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    atr = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Track daily highs/lows for divergence detection
    # We'll look for divergences over the last 3 days
    lookback_days = 3
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(adx_filter[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and trend filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check for RSI divergence (requires looking back at least 3 days)
        if i >= lookback_days * 6:  # 6 four-hour bars per day
            # Get current day's index in daily data
            current_day_idx = i // 6
            
            # Need at least 3 days of data
            if current_day_idx >= lookback_days:
                # Get RSI and price for last 3 days
                rsi_day1 = rsi_1d_aligned[(current_day_idx - 2) * 6 + 5]  # last 4h bar of day-2
                rsi_day2 = rsi_1d_aligned[(current_day_idx - 1) * 6 + 5]  # last 4h bar of day-1
                rsi_day3 = rsi_1d_aligned[current_day_idx * 6 + 5]       # last 4h bar of current day
                
                price_day1 = close[(current_day_idx - 2) * 6 + 5]
                price_day2 = close[(current_day_idx - 1) * 6 + 5]
                price_day3 = close[current_day_idx * 6 + 5]
                
                # Bullish divergence: price lower low, RSI higher low
                if (price_day3 < price_day1 and 
                    rsi_day3 > rsi_day1 and 
                    rsi_day3 < 50):  # not overbought
                    if position != 1:
                        position = 1
                        signals[i] = 0.20
                
                # Bearish divergence: price higher high, RSI lower high
                elif (price_day3 > price_day1 and 
                      rsi_day3 < rsi_day1 and 
                      rsi_day3 > 50):  # not oversold
                    if position != -1:
                        position = -1
                        signals[i] = -0.20
            else:
                # Not enough days yet, hold or flat
                if position == 1:
                    signals[i] = 0.20
                elif position == -1:
                    signals[i] = -0.20
                else:
                    signals[i] = 0.0
        else:
            # Not enough data for divergence check, hold or flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            
        # Exit on opposite divergence or extreme RSI
        if position == 1 and (rsi_1d_aligned[i] > 70 or 
                              (i >= lookback_days * 6 and 
                               price_day3 > price_day1 and 
                               rsi_day3 < rsi_day1)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_1d_aligned[i] < 30 or 
                                 (i >= lookback_days * 6 and 
                                  price_day3 < price_day1 and 
                                  rsi_day3 > rsi_day1)):
            position = 0
            signals[i] = 0.0
    
    return signals