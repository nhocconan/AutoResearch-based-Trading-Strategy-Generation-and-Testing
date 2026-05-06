#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with 1w RSI filter and volume confirmation
# Uses 1d Kaufman Adaptive Moving Average (KAMA) for trend direction
# 1w RSI(14) > 50 for long bias, < 50 for short bias to avoid counter-trend trades
# Volume spike (>2.0x 20-bar average) confirms momentum
# Fixed position size 0.25 to balance profit and fee drag; target 50-100 total trades over 4 years (12-25/year)
# Works in bull/bear: KAMA adapts to market conditions, RSI filter ensures trend alignment, volume avoids false breakouts

name = "1d_KAMA_1wRSI50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d KAMA (adaptive moving average)
    # ER = Efficiency Ratio = |net change| / sum(|changes|)
    # Smooth = ER * (fastest - slowest) + slowest
    # Alpha = Smooth^2
    def kama(close_prices, fast=2, slow=30):
        n = len(close_prices)
        kama_vals = np.full(n, np.nan)
        if n < slow + 1:
            return kama_vals
        
        # Calculate ER over slow period
        change = np.abs(np.diff(close_prices))
        for i in range(slow, n):
            net_change = np.abs(close_prices[i] - close_prices[i-slow])
            total_change = np.sum(change[i-slow+1:i+1])
            if total_change > 0:
                er = net_change / total_change
            else:
                er = 0
            fastest = 2.0 / (fast + 1)
            slowest = 2.0 / (slow + 1)
            smooth = er * (fastest - slowest) + slowest
            alpha = smooth * smooth
            
            if i == slow:
                kama_vals[i] = close_prices[i]
            else:
                kama_vals[i] = kama_vals[i-1] + alpha * (close_prices[i] - kama_vals[i-1])
        return kama_vals
    
    kama_1d = kama(close)
    
    # Calculate 1w RSI(14)
    def rsi(close_prices, period=14):
        n = len(close_prices)
        rsi_vals = np.full(n, np.nan)
        if n < period + 1:
            return rsi_vals
        
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_1w = rsi(close_1w)
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe (primary)
    kama_1d_aligned = kama_1d  # Already on 1d timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA AND RSI > 50 (bullish bias) AND volume spike
            if close[i] > kama_1d_aligned[i] and rsi_1w_aligned[i] > 50 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND RSI < 50 (bearish bias) AND volume spike
            elif close[i] < kama_1d_aligned[i] and rsi_1w_aligned[i] < 50 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA (trend reversal)
            if close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA (trend reversal)
            if close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals