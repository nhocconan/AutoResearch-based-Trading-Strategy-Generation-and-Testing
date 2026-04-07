#!/usr/bin/env python3
"""
4h_keltner_breakout_1d_trend_volume_v1
Hypothesis: On 4-hour timeframe, use Keltner channel breakouts with 1-day trend filter and volume confirmation.
Long when price breaks above Keltner upper band (EMA20 + 2*ATR(10)) with daily EMA(50) trending up and volume > 1.5x 20-period average.
Short when price breaks below Keltner lower band (EMA20 - 2*ATR(10)) with daily EMA(50) trending down and volume > 1.5x 20-period average.
Exit when price returns to the EMA20 middle band.
Designed for 15-30 trades/year to minimize fee flood while capturing strong trends with institutional validation.
Keltner channels adapt to volatility via ATR, and daily trend filter avoids counter-trend trades in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_keltner_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine daily trend direction (using EMA slope)
    daily_trend_up = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    daily_trend_down = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    for i in range(1, len(ema_50_1d_aligned)):
        if not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(ema_50_1d_aligned[i-1]):
            daily_trend_up[i] = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            daily_trend_down[i] = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
    
    # Calculate Keltner Channel (EMA20 +/- 2*ATR(10)) on 4h timeframe
    ema_period = 20
    atr_period = 10
    keltner_mult = 2.0
    
    # EMA20
    ema_20 = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).values
    
    # Keltner Bands
    keltner_upper = ema_20 + keltner_mult * atr
    keltner_lower = ema_20 - keltner_mult * atr
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(ema_period, atr_period, 20, 50), n):
        # Skip if data not available
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to EMA20 (middle band)
            if close[i] <= ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to EMA20 (middle band)
            if close[i] >= ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: price breaks above Keltner upper band with daily uptrend
                if (close[i] > keltner_upper[i] and close[i-1] <= keltner_upper[i-1] and 
                    daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Keltner lower band with daily downtrend
                elif (close[i] < keltner_lower[i] and close[i-1] >= keltner_lower[i-1] and 
                      daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals