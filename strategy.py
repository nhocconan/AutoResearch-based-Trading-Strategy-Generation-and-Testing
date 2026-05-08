#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI divergence with 1d volume confirmation and 1w ADX trend filter
# RSI divergence identifies potential reversals: bearish divergence (price up, RSI down) for shorts,
# bullish divergence (price down, RSI up) for longs. 1d volume spike confirms institutional interest.
# 1w ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranges.
# This strategy works in both bull and bear markets by combining reversal signals with trend confirmation.
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.

name = "6h_RSIDivergence_1dVolume_1wADX"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    avg_gain = rma(gain, 14)
    avg_loss = rma(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Volume spike on 1d (20-period MA)
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (vol_ma * 2.0)
    vol_spike_6h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate ADX(14) on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    tr = np.zeros_like(high_1w)
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i-1], 0)
        minus_dm[i] = max(low_1w[i-1] - low_1w[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    # Wilder smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_6h = align_htf_to_ltf(prices, df_1w, adx_strong)
    adx_weak_6h = align_htf_to_ltf(prices, df_1w, adx_weak)
    
    # RSI divergence detection (lookback 5 periods)
    def detect_divergence(rsi_arr, price_arr, lookback=5):
        bullish_div = np.zeros_like(rsi_arr, dtype=bool)
        bearish_div = np.zeros_like(rsi_arr, dtype=bool)
        
        for i in range(lookback, len(rsi_arr)):
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (price_arr[i] < price_arr[i-lookback] and 
                rsi_arr[i] > rsi_arr[i-lookback]):
                # Check if it's a meaningful divergence
                price_low = np.min(price_arr[i-lookback:i+1])
                rsi_low = np.min(rsi_arr[i-lookback:i+1])
                price_prev_low = np.min(price_arr[i-2*lookback:i-lookback+1])
                rsi_prev_low = np.min(rsi_arr[i-2*lookback:i-lookback+1])
                if price_low < price_prev_low and rsi_low > rsi_prev_low:
                    bullish_div[i] = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (price_arr[i] > price_arr[i-lookback] and 
                rsi_arr[i] < rsi_arr[i-lookback]):
                # Check if it's a meaningful divergence
                price_high = np.max(price_arr[i-lookback:i+1])
                rsi_high = np.max(rsi_arr[i-lookback:i+1])
                price_prev_high = np.max(price_arr[i-2*lookback:i-lookback+1])
                rsi_prev_high = np.max(rsi_arr[i-2*lookback:i-lookback+1])
                if price_high > price_prev_high and rsi_high < rsi_prev_high:
                    bearish_div[i] = True
        
        return bullish_div, bearish_div
    
    bullish_div, bearish_div = detect_divergence(rsi, close)
    bullish_div_6h = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bullish_div)
    bearish_div_6h = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bearish_div)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(vol_spike_6h[i]) or 
            np.isnan(adx_strong_6h[i]) or np.isnan(adx_weak_6h[i]) or
            np.isnan(bullish_div_6h[i]) or np.isnan(bearish_div_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish RSI divergence, volume spike, strong trend
            if bullish_div_6h[i] and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish RSI divergence, volume spike, strong trend
            elif bearish_div_6h[i] and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish divergence or trend weakens
            if bearish_div_6h[i] or adx_weak_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish divergence or trend weakens
            if bullish_div_6h[i] or adx_weak_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals