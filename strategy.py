#!/usr/bin/env python3
"""
4H_Keltner_Breakout_1dATRTrend_Volume
Hypothesis: Keltner Channels (20,2.0) on 4h capture volatility breakouts.
Trend filter: 1d ATR(14) normalized by price - rising ATR indicates strengthening trend.
Volume confirmation: 20-period volume EMA spike.
In bull markets, buy upper band breakouts with rising ATR trend.
In bear markets, sell lower band breakouts with rising ATR trend.
Keltner adapts to volatility better than fixed bands, reducing false breakouts in ranging markets.
Target: 20-40 trades/year, low turnover to minimize fee drag.
"""

name = "4H_Keltner_Breakout_1dATRTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Load daily data ONCE for ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Normalize ATR by price to get trend strength (rising ATR% = strengthening trend)
    atr_pct = atr14 / close_1d
    atr_pct_ma = pd.Series(atr_pct).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_trend_up = atr_pct_ma > np.roll(atr_pct_ma, 1)  # rising ATR%
    atr_trend_down = atr_pct_ma < np.roll(atr_pct_ma, 1)  # falling ATR%
    
    # Align ATR trend to 4h
    atr_trend_up_4h = align_htf_to_ltf(prices, df_1d, atr_trend_up)
    atr_trend_down_4h = align_htf_to_ltf(prices, df_1d, atr_trend_down)
    
    # Keltner Channel (20,2.0) on 4h
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_4h = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_keltner = ema20 + 2.0 * atr_4h
    lower_keltner = ema20 - 2.0 * atr_4h
    
    # Volume filter: 20-period EMA spike
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(atr_trend_up_4h[i]) or 
            np.isnan(atr_trend_down_4h[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        breakout_up = close[i] > upper_keltner[i]
        breakout_down = close[i] < lower_keltner[i]
        
        if position == 0:
            # Long: Break above upper Keltner + rising ATR trend + volume spike
            if breakout_up and atr_trend_up_4h[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Break below lower Keltner + falling ATR trend + volume spike
            elif breakout_down and atr_trend_down_4h[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Return to middle (EMA20) or ATR trend reverses
            if position == 1:
                if close[i] < ema20[i] or not atr_trend_up_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > ema20[i] or not atr_trend_down_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals