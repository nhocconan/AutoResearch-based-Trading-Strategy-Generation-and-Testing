#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + 1w RSI mean reversion + volume spike filter.
# Long when 1d KAMA slope > 0 (uptrend) AND 1w RSI < 30 (oversold) AND volume > 1.5x 20-period 1w average.
# Short when 1d KAMA slope < 0 (downtrend) AND 1w RSI > 70 (overbought) AND volume > 1.5x 20-period 1w average.
# Exit when 1d KAMA slope reverses or price crosses 1d KAMA.
# Uses discrete position size 0.25. Designed to catch trend continuations after HTF pullbacks in strong trends.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while maintaining edge.
# Works in both bull and bear markets by using KAMA for adaptive trend and RSI for mean reversion entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: KAMA (adaptive trend) ===
    close_1d = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(10).sum()
    er = change / volatility
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = [close[0]]
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    kama_slope = np.diff(kama, prepend=kama[0])  # slope = current - previous
    
    # === 1w Indicators: RSI (mean reversion) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # === 1w Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(kama_slope[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        kama_val = kama[i]
        kama_slope_val = kama_slope[i]
        rsi_val = rsi_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if KAMA slope turns negative or price crosses below KAMA
            if kama_slope_val <= 0 or price < kama_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if KAMA slope turns positive or price crosses above KAMA
            if kama_slope_val >= 0 or price > kama_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: KAMA uptrend AND 1w RSI oversold AND volume spike
            if kama_slope_val > 0 and rsi_val < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: KAMA downtrend AND 1w RSI overbought AND volume spike
            elif kama_slope_val < 0 and rsi_val > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_KAMA_1wRSI_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0