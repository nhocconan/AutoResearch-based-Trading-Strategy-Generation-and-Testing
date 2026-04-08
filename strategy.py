#!/usr/bin/env python3
"""
4h_cci_trend_reversal_v2
Hypothesis: Uses CCI(20) on 4h with 1d trend filter and volume confirmation.
Long when CCI crosses below -100 (oversold) with 1d bullish trend and volume surge.
Short when CCI crosses above +100 (overbought) with 1d bearish trend and volume surge.
Designed for 15-25 trades/year to avoid fee drag while capturing mean-reversion in trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_trend_reversal_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # CCI calculation (20-period)
    typical_price = (high + low + close) / 3
    sma_tp = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    
    cci_period = 20
    for i in range(cci_period-1, n):
        sma_tp[i] = np.mean(typical_price[i-cci_period+1:i+1])
        mean_dev = np.mean(np.abs(typical_price[i-cci_period+1:i+1] - sma_tp[i]))
        if mean_dev > 0:
            mad[i] = mean_dev
        else:
            mad[i] = 1e-10
    
    cci = np.full(n, np.nan)
    for i in range(cci_period-1, n):
        cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mad[i])
    
    # Volume filter: 1.8x 20-period average (approx 3.3 days)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.8 * vol_ma[i]
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_50_1d[i] = close_1d[i]
        elif not np.isnan(close_1d[i]):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
        else:
            ema_50_1d[i] = ema_50_1d[i-1]
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_1d = close_1d > ema_50_1d
    bearish_1d = close_1d < ema_50_1d
    
    # Align daily trend to 4h timeframe
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d.astype(float))
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(cci_period, vol_ma_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci[i]) or np.isnan(bullish_1d_aligned[i]) or 
            np.isnan(bearish_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses above 0 or trend turns bearish
            if i > 0 and cci[i-1] <= 0 and cci[i] > 0:
                position = 0
                signals[i] = 0.0
            elif bearish_1d_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses below 0 or trend turns bullish
            if i > 0 and cci[i-1] >= 0 and cci[i] < 0:
                position = 0
                signals[i] = 0.0
            elif bullish_1d_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI crosses below -100, bullish trend, volume surge
            if i > 0 and cci[i-1] > -100 and cci[i] <= -100:
                if bullish_1d_aligned[i] > 0.5 and vol_surge[i]:
                    position = 1
                    signals[i] = 0.25
            # Short entry: CCI crosses above +100, bearish trend, volume surge
            elif i > 0 and cci[i-1] < 100 and cci[i] >= 100:
                if bearish_1d_aligned[i] > 0.5 and vol_surge[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals