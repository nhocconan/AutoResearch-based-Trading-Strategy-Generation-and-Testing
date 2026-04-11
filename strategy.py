#!/usr/bin/env python3
# 1h_4h_1d_rsi_volume_breakout_v1
# Strategy: 1h RSI mean reversion with 4h/1d trend filter and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: In ranging markets, RSI extremes (<30 or >70) with volume spikes offer mean reversion opportunities.
# In trending markets, 4h/1d trend filters prevent counter-trend trades. Works in both bull/bear by combining
# mean reversion in ranges with trend-following in strong moves. Low trade frequency via strict 4h/1d trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_volume_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 14-period RSI on 1h
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA200 for long-term trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_avg_20[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Trend filters
        # 4h trend: price above/below EMA50
        trend_4h_bullish = close[i] > ema_50_4h_aligned[i]
        trend_4h_bearish = close[i] < ema_50_4h_aligned[i]
        # 1d trend: price above/below EMA200 (stronger filter)
        trend_1d_bullish = close[i] > ema_200_1d_aligned[i]
        trend_1d_bearish = close[i] < ema_200_1d_aligned[i]
        
        # Entry conditions
        # Long: RSI oversold + volume spike + 4h bullish OR 1d bullish (avoid fighting strong trends)
        if rsi_oversold and vol_confirm and (trend_4h_bullish or trend_1d_bullish) and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: RSI overbought + volume spike + 4h bearish OR 1d bearish
        elif rsi_overbought and vol_confirm and (trend_4h_bearish or trend_1d_bearish) and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi[i] > 40:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi[i] < 60:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals