#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with 1w RSI regime filter and volume confirmation.
# Uses 1d Kaufman Adaptive Moving Average (KAMA) for trend identification.
# Uses 1w RSI to filter regime: only take longs when 1w RSI > 50 (bullish regime),
# only shorts when 1w RSI < 50 (bearish regime).
# Adds volume confirmation: current volume > 1.5 * 20-period average volume.
# Discrete sizing 0.25 to balance return and drawdown. Target: 10-20 trades/year.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# 1w RSI regime filter ensures we trade with the higher timeframe momentum.
# Volume confirmation ensures trades occur with participation.
# Works in bull markets (follow KAMA up when 1w RSI > 50) and bear markets
# (follow KAMA down when 1w RSI < 50) by aligning with weekly structure.

name = "1d_KAMA_1wRSI_Regime_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data for KAMA calculation (same timeframe as prices)
    df_1d = prices  # prices is already 1d data
    
    # Load 1w data ONCE before loop for RSI regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio = |close - close[10]| / sum(|close - close[-1]|) over 10 periods
    # Smoothest = 2/(2+1) = 0.666..., Fastest = 2/(30+1) = 0.0645
    # SC = [ER * (fastest - smoothest) + smoothest]^2
    # KAMA = prev_KAMA + SC * (price - prev_KAMA)
    
    def calculate_kama(close_prices, period=10, fast=2, slow=30):
        """Calculate Kaufman Adaptive Moving Average"""
        if len(close_prices) < period + 1:
            return np.full_like(close_prices, np.nan)
        
        # Calculate directional change
        directional_change = np.abs(close_prices[period:] - close_prices[:-period])
        
        # Calculate total absolute price change
        abs_price_changes = np.abs(np.diff(close_prices))
        total_change = np.zeros_like(close_prices)
        for i in range(period, len(close_prices)):
            total_change[i] = np.sum(abs_price_changes[i-period:i])
        
        # Avoid division by zero
        efficiency_ratio = np.zeros_like(close_prices)
        mask = total_change[period:] != 0
        efficiency_ratio[period:] = np.where(
            mask,
            directional_change / total_change[period:],
            0
        )
        
        # Calculate smoothing constant
        fastest_sc = 2.0 / (fast + 1)
        slowest_sc = 2.0 / (slow + 1)
        sc = (efficiency_ratio * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # Calculate KAMA
        kama = np.full_like(close_prices, np.nan)
        kama[period] = close_prices[period]  # Start with first available close
        
        for i in range(period + 1, len(close_prices)):
            if not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
            else:
                kama[i] = close_prices[i]
        
        return kama
    
    # Calculate 1d KAMA
    kama_1d = calculate_kama(close, period=10, fast=2, slow=30)
    
    # Calculate 1w RSI for regime filter
    close_1w = df_1w['close'].values
    if len(close_1w) < 14:
        rsi_1w = np.full_like(close_1w, np.nan)
    else:
        # Calculate RSI using Wilder's smoothing
        delta = np.diff(close_1w)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing (alpha = 1/period)
        avg_gain = np.full_like(close_1w, np.nan)
        avg_loss = np.full_like(close_1w, np.nan)
        
        # First average is simple average
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        # Subsequent values using Wilder's smoothing
        for i in range(14, len(close_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        # Calculate RSI
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1w = 100 - (100 / (1 + rs))
        # Set first 13 values to NaN
        rsi_1w[:13] = np.nan
    
    # Align 1w RSI to 1d
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.zeros_like(volume)
    mask = vol_ma_20 > 0
    volume_ratio[mask] = volume[mask] / vol_ma_20[mask]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(kama_1d[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_kama = kama_1d[i]
        curr_rsi = rsi_1w_aligned[i]
        curr_vol_ratio = volume_ratio[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Price above KAMA AND 1w RSI > 50 (bullish regime) AND volume confirmation
            if (curr_close > curr_kama and 
                curr_rsi > 50 and 
                curr_vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA AND 1w RSI < 50 (bearish regime) AND volume confirmation
            elif (curr_close < curr_kama and 
                  curr_rsi < 50 and 
                  curr_vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below KAMA OR 1w RSI < 40 (regime change to bearish)
            if (curr_close < curr_kama or 
                curr_rsi < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above KAMA OR 1w RSI > 60 (regime change to bullish)
            if (curr_close > curr_kama or 
                curr_rsi > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals