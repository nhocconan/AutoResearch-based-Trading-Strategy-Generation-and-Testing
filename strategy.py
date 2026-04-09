#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + chop regime filter (CHOP < 61.8 = trending)
# KAMA adapts to market efficiency - follows trend in trending markets, avoids whipsaws in ranging
# RSI(14) confirms momentum strength (avoid overbought/oversold extremes)
# Chop regime filter: only trade when CHOP < 61.8 (not strongly ranging) to reduce false signals
# Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25
# Works in bull/bear: KAMA follows trends, chop filter prevents trading in strong ranging markets

name = "1d_kama_rsi_chop_v1"
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
    
    # Load 1w data ONCE before loop for regime context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Handle first period values
        er = np.full_like(change, np.nan, dtype=np.float64)
        er[period-1:] = change / np.where(volatility[period-1:] == 0, 1, volatility[period-1:])
        er = np.concatenate([np.full(period, np.nan), er])
        
        # Smoothing constants
        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama_vals = np.full_like(close, np.nan, dtype=np.float64)
        kama_vals[period] = close[period]  # Start with first close after period
        for i in range(period + 1, len(close)):
            if np.isnan(kama_vals[i-1]):
                kama_vals[i] = close[i]
            else:
                kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate RSI(14)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan, dtype=np.float64)
        avg_loss = np.full_like(close, np.nan, dtype=np.float64)
        
        # First average
        avg_gain[period] = np.nanmean(gain[:period])
        avg_loss[period] = np.nanmean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Calculate 1w Chopiness Index for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Chopiness Index
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1w - ll_1w
    chop_1w = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align 1w indicators to 1d timeframe (wait for 1w bar close)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when not strongly ranging (CHOP < 61.8)
        not_strongly_ranging = chop_1w_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below KAMA OR RSI < 30 (momentum loss) OR strong ranging
            if close[i] < kama_vals[i] or rsi_vals[i] < 30 or not not_strongly_ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above KAMA OR RSI > 70 (momentum loss) OR strong ranging
            if close[i] > kama_vals[i] or rsi_vals[i] > 70 or not not_strongly_ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: only in non-strongly-ranging regime
            if not_strongly_ranging:
                # Long: price above KAMA AND RSI > 50 (bullish momentum)
                if close[i] > kama_vals[i] and rsi_vals[i] > 50:
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA AND RSI < 50 (bearish momentum)
                elif close[i] < kama_vals[i] and rsi_vals[i] < 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals