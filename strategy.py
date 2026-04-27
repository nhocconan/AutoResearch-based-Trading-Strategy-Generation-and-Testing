#!/usr/bin/env python3
"""
1d_Keltner_Breakout_1wTrend_VolumeFilter
Hypothesis: Combines Keltner channel breakouts with weekly trend filter and volume confirmation.
Designed for daily timeframe to capture medium-term trends with low trade frequency.
Keltner channels (EMA + ATR) adapt to volatility, making them effective in both trending and ranging markets.
Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
Volume confirmation filters out low-conviction breakouts.
Target: 10-25 trades per year to minimize fee drag in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily Keltner Channel (20-period EMA, 2.0 ATR multiplier)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    keltner_upper = ema_20 + (2.0 * atr)
    keltner_lower = ema_20 - (2.0 * atr)
    
    # Weekly trend filter: 50-period EMA
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = keltner_upper[i]
        lower = keltner_lower[i]
        weekly_trend = ema_50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner band with uptrend and volume
            if close_val > upper and close_val > weekly_trend and vol_conf:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Keltner band with downtrend and volume
            elif close_val < lower and close_val < weekly_trend and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA(20) (trend reversal signal)
            if close_val < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA(20) (trend reversal signal)
            if close_val > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Keltner_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0