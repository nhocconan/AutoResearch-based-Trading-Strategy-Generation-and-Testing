#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Ichimoku Cloud with daily trend filter and volume confirmation.
# Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h data.
# Daily trend filter (price vs EMA50) ensures trades align with higher timeframe bias.
# Volume confirmation (current volume > 1.5x 20-period average) filters low-quality breakouts.
# Designed for 6h timeframe to target 50-150 trades over 4 years.
# Works in bull/bear markets via daily EMA trend bias and cloud breakout logic.

name = "6h_ichimoku_daily_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily closes
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / 51) + (ema_50_1d[i-1] * 49 / 51)
    
    # Align EMA50 to 6h timeframe (shifted by 1 daily bar for no look-ahead)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = np.full(n, np.nan)
    for i in range(8, n):
        tenkan_sen[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = np.full(n, np.nan)
    for i in range(25, n):
        kijun_sen[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = np.full(n, np.nan)
    for i in range(25, n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = np.full(n, np.nan)
    for i in range(51, n):
        senkou_span_b[i] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):  # Start after Senkou Span B is available
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or 
            np.isnan(senkou_span_b[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        if np.isnan(vol_ma):
            volume_filter = False
        else:
            volume_filter = volume[i] > vol_ma * 1.5
        
        # Trend bias: daily EMA50
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        # Cloud breakout conditions
        # Cloud top = max(Senkou Span A, Senkou Span B)
        # Cloud bottom = min(Senkou Span A, Senkou Span B)
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        # Breakout above cloud (bullish)
        breakout_above = close[i] > cloud_top and close[i-1] <= cloud_top
        # Breakout below cloud (bearish)
        breakout_below = close[i] < cloud_bottom and close[i-1] >= cloud_bottom
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price re-enters cloud or stoploss (2x ATR approximation)
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] < cloud_top or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price re-enters cloud or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] > cloud_bottom or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of daily trend with volume confirmation
            if volume_filter:
                # Long: breakout above cloud in uptrend
                if breakout_above and bullish_bias:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakout below cloud in downtrend
                elif breakout_below and bearish_bias:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals