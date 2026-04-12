#!/usr/bin/env python3
"""
4h_12h_CCI_Trend_Filter_v1
Hypothesis: Use 12-hour CCI(20) to filter 4-hour breakout signals. Long when price breaks above upper Keltner Channel (EMA20 + 2*ATR(10)) with 12h CCI > 100 and volume > 1.5x average. Short when price breaks below lower Keltner Channel with 12h CCI < -100 and volume > 1.5x average. Exit when price returns to EMA20 or CCI crosses zero. Designed for 20-40 trades/year with strong trend filtering to avoid whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_CCI_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR CCI TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate CCI(20)
    typical_price = (high_12h + low_12h + close_12h) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    cci = np.where(mad != 0, cci, 0.0)
    
    # Align CCI to 4h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_12h, cci)
    
    # === 4H INDICATORS: KELTNER CHANNEL ===
    # EMA20 for Keltner middle
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) for Keltner width
    atr_period = 10
    tr_4h = np.zeros_like(high)
    for i in range(1, len(high)):
        tr_4h[i] = max(high[i] - low[i], 
                      abs(high[i] - close[i-1]), 
                      abs(low[i] - close[i-1]))
    atr_10 = pd.Series(tr_4h).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Keltner Bands
    kc_upper = ema20 + 2 * atr_10
    kc_lower = ema20 - 2 * atr_10
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if not ready
        if (np.isnan(ema20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(cci_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: price breaks above Keltner Upper with strong CCI and volume
        long_signal = (close[i] > kc_upper[i] and 
                      cci_aligned[i] > 100 and 
                      strong_volume)
        
        # Short: price breaks below Keltner Lower with strong CCI and volume
        short_signal = (close[i] < kc_lower[i] and 
                       cci_aligned[i] < -100 and 
                       strong_volume)
        
        # Exit: price returns to middle (EMA20) or CCI crosses zero
        exit_long = (position == 1 and 
                    (close[i] < ema20[i] or cci_aligned[i] < 0))
        exit_short = (position == -1 and 
                     (close[i] > ema20[i] or cci_aligned[i] > 0))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals