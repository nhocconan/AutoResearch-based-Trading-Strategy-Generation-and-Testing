#!/usr/bin/env python3
# 1h_rsi_divergence_4h_trend_volume_v1
# Hypothesis: On 1h timeframe, use RSI divergence (bullish/bearish) with 4h trend filter and volume confirmation.
# Bullish divergence: price makes lower low while RSI makes higher low → long when price closes above prior swing high.
# Bearish divergence: price makes higher high while RSI makes lower high → short when price closes below prior swing low.
# 4h trend filter: price above/below 4h EMA50 to align with higher timeframe trend.
# Volume confirmation: current volume > 1.3x 20-period average to avoid low-volume false signals.
# Targets 15-37 trades/year by requiring multiple confluence factors (divergence + trend + volume).
# Works in bull markets via bullish divergences in uptrends and bear markets via bearish divergences in downtrends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_divergence_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI calculation (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = np.nan
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Swing points for divergence detection (5-period lookback)
    def find_swing_points(arr, lookback=5):
        highs = np.full_like(arr, np.nan)
        lows = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr) - lookback):
            if arr[i] == np.max(arr[i-lookback:i+lookback+1]):
                highs[i] = arr[i]
            if arr[i] == np.min(arr[i-lookback:i+lookback+1]):
                lows[i] = arr[i]
        return highs, lows
    
    price_highs, price_lows = find_swing_points(close, 5)
    rsi_highs, rsi_lows = find_swing_points(rsi, 5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h EMA50 or RSI > 70 (overbought)
            if close[i] < ema_50_4h_aligned[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h EMA50 or RSI < 30 (oversold)
            if close[i] > ema_50_4h_aligned[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.3x average volume
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # 4h trend filter
            uptrend_4h = close[i] > ema_50_4h_aligned[i]
            downtrend_4h = close[i] < ema_50_4h_aligned[i]
            
            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = False
            if i >= 20:  # Need sufficient history
                # Find last two price lows
                price_low_indices = []
                for j in range(max(0, i-30), i):
                    if not np.isnan(price_lows[j]):
                        price_low_indices.append(j)
                if len(price_low_indices) >= 2:
                    low1_idx, low2_idx = price_low_indices[-2], price_low_indices[-1]
                    if (low2_idx > low1_idx and 
                        close[low2_idx] < close[low1_idx] and  # lower low in price
                        not np.isnan(rsi_lows[low2_idx]) and not np.isnan(rsi_lows[low1_idx]) and
                        rsi_lows[low2_idx] > rsi_lows[low1_idx]):  # higher low in RSI
                        bullish_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = False
            if i >= 20:  # Need sufficient history
                # Find last two price highs
                price_high_indices = []
                for j in range(max(0, i-30), i):
                    if not np.isnan(price_highs[j]):
                        price_high_indices.append(j)
                if len(price_high_indices) >= 2:
                    high1_idx, high2_idx = price_high_indices[-2], price_high_indices[-1]
                    if (high2_idx > high1_idx and 
                        close[high2_idx] > close[high1_idx] and  # higher high in price
                        not np.isnan(rsi_highs[high2_idx]) and not np.isnan(rsi_highs[high1_idx]) and
                        rsi_highs[high2_idx] < rsi_highs[high1_idx]):  # lower high in RSI
                        bearish_div = True
            
            # Long entry: bullish divergence + uptrend + volume
            if bullish_div and uptrend_4h and volume_ok:
                position = 1
                signals[i] = 0.20
            # Short entry: bearish divergence + downtrend + volume
            elif bearish_div and downtrend_4h and volume_ok:
                position = -1
                signals[i] = -0.20
    
    return signals