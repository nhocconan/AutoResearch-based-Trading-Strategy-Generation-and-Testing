#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_V4
Hypothesis: On daily timeframe, KAMA direction filter + RSI(14) mean reversion + Choppiness Index regime filter.
Long when KAMA trending up, RSI < 40, and choppy market (CHOP > 61.8). Short when KAMA trending down, RSI > 60, and choppy market.
Uses volume confirmation (>1.5x avg volume) to avoid false signals. Discrete position sizing (0.25) to minimize fee churn.
Designed to work in ranging markets (chop filter) with mean reversion entries, avoiding strong trends that cause whipsaw.
Target trades: 15-25/year to stay within fee drag limits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter (optional, but can help avoid counter-trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate KAMA on close prices
    def calculate_kama(close_arr, period=10, fast=2, slow=30):
        """Kaufman Adaptive Moving Average"""
        n = len(close_arr)
        kama = np.full(n, np.nan)
        if n < period:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.diff(close_arr, n=period))
        volatility = np.sum(np.abs(np.diff(close_arr)), axis=0)
        er = np.zeros(n)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        
        # Smoothing constant
        sc = np.square(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))
        
        # Initialize KAMA
        kama[period-1] = close_arr[period-1]
        
        # Calculate KAMA
        for i in range(period, n):
            kama[i] = kama[i-1] + sc[i] * (close_arr[i] - kama[i-1])
        
        return kama
    
    # Calculate RSI
    def calculate_rsi(close_arr, period=14):
        """Relative Strength Index"""
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_arr)
        avg_loss = np.zeros_like(close_arr)
        
        # Wilder's smoothing
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(close_arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Choppiness Index - measures if market is choppy (trending) or ranging"""
        n = len(close_arr)
        chop = np.full(n, np.nan)
        if n < period:
            return chop
        
        # True Range
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Sum of True Range over period
        atr_sum = np.zeros(n)
        for i in range(period, n):
            atr_sum[i] = np.nansum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros(n)
        ll = np.zeros(n)
        for i in range(period, n):
            hh[i] = np.nanmax(high_arr[i-period+1:i+1])
            ll[i] = np.nanmin(low_arr[i-period+1:i+1])
        
        # Chop calculation
        for i in range(period, n):
            if atr_sum[i] > 0 and (hh[i] - ll[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_chop(high, low, close, period=14)
    
    # Weekly trend filter (optional)
    if len(df_1w) >= 10:
        ema_10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
        ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
        weekly_uptrend = ema_10_1w_aligned > 0  # Will be refined below
    else:
        ema_10_1w_aligned = np.full(n, np.nan)
        weekly_uptrend = np.ones(n, dtype=bool)  # Default to true if no weekly data
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30 for KAMA/RSI/CHOP, 20 for volume)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        # Weekly trend (if available)
        weekly_trend_up = True
        weekly_trend_down = True
        if not np.isnan(ema_10_1w_aligned[i]) and i >= 10:
            weekly_trend_up = close_val > ema_10_1w_aligned[i]
            weekly_trend_down = close_val < ema_10_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(avg_vol)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # KAMA direction: price above/below KAMA indicates trend
        kama_up = close_val > kama_val
        kama_down = close_val < kama_val
        
        # RSI conditions for mean reversion
        rsi_oversold = rsi_val < 40
        rsi_overbought = rsi_val > 60
        
        # Choppiness regime: choppy market (mean reversion favorable)
        choppy_market = chop_val > 61.8
        
        # Long logic: KAMA up, RSI oversold, choppy market, volume confirmation
        long_condition = (kama_up and rsi_oversold and choppy_market and 
                         volume_confirmed and weekly_trend_up)
        
        # Short logic: KAMA down, RSI overbought, choppy market, volume confirmation
        short_condition = (kama_down and rsi_overbought and choppy_market and 
                          volume_confirmed and weekly_trend_down)
        
        # Exit logic: reverse of entry conditions or trend change
        long_exit = (position == 1 and 
                    (not kama_up or rsi_val > 50 or not choppy_market))
        short_exit = (position == -1 and 
                     (not kama_down or rsi_val < 50 or not choppy_market))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_V4"
timeframe = "1d"
leverage = 1.0