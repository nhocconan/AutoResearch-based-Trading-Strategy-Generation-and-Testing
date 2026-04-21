#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend following with 1d RSI filter and volume confirmation.
# Uses Kaufman Adaptive Moving Average to reduce whipsaw in choppy markets.
# RSI filter prevents entries during overbought/oversold conditions.
# Volume confirmation ensures institutional participation.
# Target: 20-30 trades/year by requiring KAMA trend alignment, RSI filter, and volume spike.
# Works in bull markets by following trends, in bear markets by avoiding false signals during low volume.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on daily close
    close_d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_d, prepend=close_d[0]))
    volatility = np.abs(np.diff(close_d))
    er = np.zeros_like(close_d)
    er[1:] = change[1:] / (np.sum(volatility[np.arange(1, len(close_d))[:, None] <= np.arange(1, len(close_d))], axis=1) + 1e-10)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_d)
    kama[0] = close_d[0]
    for i in range(1, len(close_d)):
        kama[i] = kama[i-1] + sc[i] * (close_d[i] - kama[i-1])
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_d, prepend=close_d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load weekly volume for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    vol_w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_w).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate 12h price for comparison
    close_12h = prices['close'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = close_12h[i]
        vol_current = align_htf_to_ltf(prices, df_1w, vol_w)[i]  # weekly volume aligned to 12h
        
        # Trend filter: price relative to KAMA
        above_kama = price_close > kama_1d_aligned[i]
        below_kama = price_close < kama_1d_aligned[i]
        
        # RSI filter: avoid extreme levels
        rsi_ok = (rsi_1d_aligned[i] > 30) & (rsi_1d_aligned[i] < 70)
        
        # Volume confirmation: current volume > 1.3x 20-week average
        volume_confirm = vol_current > 1.3 * vol_ma_20_1w_aligned[i]
        
        if position == 0:
            # Enter long when price above KAMA, RSI not overbought, and volume spike
            if above_kama and rsi_ok and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short when price below KAMA, RSI not oversold, and volume spike
            elif below_kama and rsi_ok and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: opposite KAMA cross or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below KAMA or volume drops
                if price_close < kama_1d_aligned[i]:
                    exit_signal = True
                elif vol_current < vol_ma_20_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above KAMA or volume drops
                if price_close > kama_1d_aligned[i]:
                    exit_signal = True
                elif vol_current < vol_ma_20_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_Trend_RSIFilter_Volume"
timeframe = "12h"
leverage = 1.0