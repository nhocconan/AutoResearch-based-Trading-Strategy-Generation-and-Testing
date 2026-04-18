#!/usr/bin/env python3
"""
1h_Pullback_RSI_Volume_Trend_v1
Strategy: 1h pullback to EMA21 in uptrend/downtrend with RSI filter and volume confirmation.
Long: Uptrend (EMA21 > EMA50) + price pulls back to EMA21 + RSI < 40 + volume > 1.5x avg.
Short: Downtrend (EMA21 < EMA50) + price pulls back to EMA21 + RSI > 60 + volume > 1.5x avg.
Designed for 1h timeframe: ~20-40 trades/year per symbol (80-160 total over 4 years).
Works in bull via trend-following pulls back; works in bear via shorting bounces in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (more stable than 1h EMA cross)
    df_4h = get_htf_data(prices, '4h')
    
    close_4h = df_4h['close'].values
    # 4h EMA21 and EMA50 for trend filter
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h indicators for entry timing
    close_s = pd.Series(close)
    ema_21_1h = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_1h = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_21_1h[i]) or np.isnan(ema_50_1h[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend from 4h EMA (more reliable)
        uptrend = ema_21_4h_aligned[i] > ema_50_4h_aligned[i]
        downtrend = ema_21_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Pullback to 1h EMA21 with RSI filter
        near_ema21 = abs(close[i] - ema_21_1h[i]) < (0.005 * ema_21_1h[i])  # within 0.5%
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        if position == 0:
            # Long: uptrend + pullback to EMA21 + oversold RSI + volume
            if uptrend and near_ema21 and rsi_oversold and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + pullback to EMA21 + overbought RSI + volume
            elif downtrend and near_ema21 and rsi_overbought and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend change, RSI overbought, or price breaks below EMA21
            if not uptrend or rsi[i] > 70 or close[i] < ema_21_1h[i]:
                signals[i] = 0.0  # exit to flat
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend change, RSI oversold, or price breaks above EMA21
            if not downtrend or rsi[i] < 30 or close[i] > ema_21_1h[i]:
                signals[i] = 0.0  # exit to flat
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Pullback_RSI_Volume_Trend_v1"
timeframe = "1h"
leverage = 1.0