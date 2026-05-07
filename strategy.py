#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction on 1d timeframe, combined with RSI(14) and Choppiness Index (CHOP) regime filter to enter in strong trends and avoid whipsaws. Choppiness Index > 61.8 indicates ranging (mean-revert), < 38.2 indicates trending (trend-follow). This combination should work in both bull (trend-following) and bear (mean-reversion in ranges) markets while keeping trades low via regime filtering.
"""
name = "1d_KAMA_Direction_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    
    # === INDICATOR CALCULATIONS ON 1D DATA ===
    # Get 1d data for all calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA ( Kaufman Adaptive Moving Average )
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 0 else np.abs(np.diff(close)).sum()
        # Handle 1D case
        if len(change.shape) == 1:
            volatility = np.abs(np.diff(close))
            er = np.zeros_like(close)
            for i in range(1, len(close)):
                if np.sum(volatility[max(0, i-length+1):i+1]) > 0:
                    er[i] = np.abs(close[i] - close[max(0, i-length+1)]) / np.sum(volatility[max(0, i-length+1):i+1])
                else:
                    er[i] = 0
        else:
            er = np.zeros_like(close)
            for i in range(len(close)):
                start_idx = max(0, i - length + 1)
                if np.sum(volatility[start_idx:i+1]) > 0:
                    er[i] = np.abs(close[i] - close[start_idx]) / np.sum(volatility[start_idx:i+1])
                else:
                    er[i] = 0
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI calculation
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Wilder's smoothing
        avg_gain[length-1] = np.mean(gain[1:length]) if length > 1 else 0
        avg_loss[length-1] = np.mean(loss[1:length]) if length > 1 else 0
        
        for i in range(length, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index calculation
    def calculate_chop(high, low, close, length=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First period
        
        # Sum of True Range over period
        atr_sum = np.zeros_like(close)
        for i in range(length-1, len(close)):
            atr_sum[i] = np.sum(tr[max(0, i-length+1):i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(length-1, len(close)):
            hh[i] = np.max(high[max(0, i-length+1):i+1])
            ll[i] = np.min(low[max(0, i-length+1):i+1])
        
        # Choppiness Index
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if atr_sum[i] > 0 and (hh[i] - ll[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(length)
            else:
                chop[i] = 50  # Neutral
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close_1d, length=10, fast=2, slow=30)
    rsi = calculate_rsi(close_1d, length=14)
    chop = calculate_chop(high_1d, low_1d, close_1d, length=14)
    
    # Align to lower timeframe (1d to 1d is identity, but we keep for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Optional: 1-week trend filter (only use if 1w data available)
    if len(df_1w) >= 10:
        close_1w = df_1w['close'].values
        # Simple trend: price above/below 20-period EMA on weekly
        ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
        ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
        # Trend filter: only take longs in weekly uptrend, shorts in weekly downtrend
        weekly_uptrend = close_1w[-1] > ema_20_1w[-1] if len(close_1w) > 0 else False
        weekly_downtrend = close_1w[-1] < ema_20_1w[-1] if len(close_1w) > 0 else False
    else:
        ema_20_1w_aligned = np.full(n, np.nan)
        weekly_uptrend = True
        weekly_downtrend = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: Choppiness Index
        # CHOP > 61.8 = ranging (mean-revert), CHOP < 38.2 = trending (trend-follow)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # LONG CONDITIONS
            # In trending markets: trend follow (price > KAMA)
            # In ranging markets: mean revert (RSI < 30 and price > KAMA for bullish bias)
            long_condition = False
            if is_trending:
                # Trend following: price above KAMA
                long_condition = close[i] > kama_aligned[i]
            elif is_ranging:
                # Mean reversion: oversold RSI with bullish bias
                long_condition = (rsi_aligned[i] < 30) and (close[i] > kama_aligned[i])
            
            # SHORT CONDITIONS
            # In trending markets: trend follow (price < KAMA)
            # In ranging markets: mean revert (RSI > 70 and price < KAMA for bearish bias)
            short_condition = False
            if is_trending:
                # Trend following: price below KAMA
                short_condition = close[i] < kama_aligned[i]
            elif is_ranging:
                # Mean reversion: overbought RSI with bearish bias
                short_condition = (rsi_aligned[i] > 70) and (close[i] < kama_aligned[i])
            
            # Apply weekly trend filter (optional)
            if len(df_1w) >= 10 and not np.isnan(ema_20_1w_aligned[i]):
                if weekly_uptrend and not weekly_downtrend:
                    # Only allow longs in weekly uptrend
                    short_condition = False
                elif weekly_downtrend and not weekly_uptrend:
                    # Only allow shorts in weekly downtrend
                    long_condition = False
            
            # Enter long
            if long_condition:
                signals[i] = 0.25
                position = 1
            # Enter short
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # EXIT CONDITIONS
            if position == 1:  # Long position
                # Exit when: price crosses below KAMA OR RSI becomes overbought in ranging market
                if close[i] < kama_aligned[i] or (is_ranging and rsi_aligned[i] > 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit when: price crosses above KAMA OR RSI becomes oversold in ranging market
                if close[i] > kama_aligned[i] or (is_ranging and rsi_aligned[i] < 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals