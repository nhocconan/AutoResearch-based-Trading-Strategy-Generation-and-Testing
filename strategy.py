#!/usr/bin/env python3
# 1h_ema_trend_macd_volume_v1
# Hypothesis: 1h strategy using EMA trend filter (4h EMA50) for direction, MACD histogram for momentum timing, and volume confirmation.
# Long when 4h EMA50 rising, MACD histogram crosses above zero, and volume > 1.5x 20-period average.
# Short when 4h EMA50 falling, MACD histogram crosses below zero, and volume > 1.5x 20-period average.
# Exit when MACD histogram crosses back through zero in opposite direction.
# Uses 4h/1d for signal direction, 1h only for entry timing to reduce trade frequency.
# Session filter (08-20 UTC) to avoid low-liquidity periods.
# Discrete position sizing (0.20) to minimize fee churn.
# Target: 15-37 trades/year (60-150 total over 4 years) on BTC/ETH/SOL to avoid fee drag.
# Works in both bull and bear markets: EMA filter adapts to trend, MACD captures momentum shifts, volume confirms conviction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_trend_macd_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop (4h for trend, 1d for higher context)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA50 for trend direction
    close_4h = pd.Series(df_4h['close'].values)
    ema_4h_50 = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # 1d EMA200 for higher timeframe filter (optional trend strength)
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d_200 = close_1d.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # MACD histogram (12,26,9)
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=12, min_periods=12, adjust=False).mean()
    ema_slow = close_s.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    macd_hist_values = macd_hist.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(ema_1d_200_aligned[i]) or
            np.isnan(macd_hist_values[i]) or np.isnan(macd_hist_values[i-1]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filters: 4h EMA50 slope and price vs 1d EMA200
        ema_4h_rising = ema_4h_50_aligned[i] > ema_4h_50_aligned[i-1]
        ema_4h_falling = ema_4h_50_aligned[i] < ema_4h_50_aligned[i-1]
        price_above_1d_ema = close[i] > ema_1d_200_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_200_aligned[i]
        
        if position == 1:  # Long position
            # Exit: MACD histogram crosses below zero OR trend breaks
            if (macd_hist_values[i] < 0 and macd_hist_values[i-1] >= 0) or not (ema_4h_rising and price_above_1d_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: MACD histogram crosses above zero OR trend breaks
            if (macd_hist_values[i] > 0 and macd_hist_values[i-1] <= 0) or not (ema_4h_falling and price_below_1d_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for MACD zero-cross with volume and trend confirmation
            bullish_setup = (macd_hist_values[i] > 0 and macd_hist_values[i-1] <= 0) and volume_confirmed
            bearish_setup = (macd_hist_values[i] < 0 and macd_hist_values[i-1] >= 0) and volume_confirmed
            
            if bullish_setup and ema_4h_rising and price_above_1d_ema:
                position = 1
                signals[i] = 0.20
            elif bearish_setup and ema_4h_falling and price_below_1d_ema:
                position = -1
                signals[i] = -0.20
    
    return signals