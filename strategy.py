#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Confirmation
# Hypothesis: 12h trend following using KAMA direction with RSI momentum confirmation and volume filter.
# Uses 1d EMA34 for trend direction, KAMA(14,2,30) for entry timing, RSI(14) for momentum filter, and volume spike for confirmation.
# Designed for low trade frequency (12-37/year) to minimize fee drag while capturing trends in both bull and bear markets.

name = "12h_KAMA_Trend_RSI_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily trend filter: EMA34 ---
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- KAMA (14,2,30) for entry timing ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- RSI(14) for momentum ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # --- Volume confirmation (2.0x 24-period average) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: ensure we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs daily EMA34
        bullish_trend = close[i] > ema_34_1d_aligned[i]
        bearish_trend = close[i] < ema_34_1d_aligned[i]
        
        # KAMA direction: price vs KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI momentum: avoid extremes, look for momentum
        rsi_bullish = 50 < rsi[i] < 70
        rsi_bearish = 30 < rsi[i] < 50
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA and daily EMA34, bullish RSI momentum, volume surge
            if price_above_kama and bullish_trend and rsi_bullish and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and daily EMA34, bearish RSI momentum, volume surge
            elif price_below_kama and bearish_trend and rsi_bearish and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit: price crosses below KAMA or RSI turns bearish
                if close[i] < kama[i] or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: price crosses above KAMA or RSI turns bullish
                if close[i] > kama[i] or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals