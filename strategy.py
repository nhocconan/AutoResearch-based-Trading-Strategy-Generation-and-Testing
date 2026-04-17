#!/usr/bin/env python3
"""
Hypothesis:
This strategy uses 12-hour timeframe with 1-week moving average for trend direction,
1-day RSI for overbought/oversold conditions, and volume confirmation for entry timing.
It aims to capture medium-term reversals in both bull and bear markets by combining
trend-following (weekly MA) with mean-reversion (daily RSI). The 12h timeframe
reduces trade frequency to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-week EMA (50-period) for trend direction ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1-day RSI (14-period) for overbought/oversold ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1-day volume spike (vs 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup for weekly EMA
        # Skip if any data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_spike = volume_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 1.3
        
        # Trend filter: price above/below weekly EMA
        price_above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike:
                # Long: price above weekly EMA AND RSI oversold
                if price_above_weekly_ema and rsi_oversold:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: price below weekly EMA AND RSI overbought
                elif price_below_weekly_ema and rsi_overbought:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long if price crosses below weekly EMA or RSI becomes overbought
            if close[i] < ema_50_1w_aligned[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price crosses above weekly EMA or RSI becomes oversold
            if close[i] > ema_50_1w_aligned[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA50_DailyRSI_VolumeSpike"
timeframe = "12h"
leverage = 1.0