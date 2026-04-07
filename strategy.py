#!/usr/bin/env python3
"""
4h_keltner_breakout_1d_trend_volume_v1
Hypothesis: On 4-hour timeframe, use Keltner Channel breakouts with 1-day trend filter and volume confirmation.
Long when price breaks above Keltner upper band (EMA10 + 2*ATR) with daily EMA(50) trending up and volume > 1.5x 20-period average.
Short when price breaks below Keltner lower band (EMA10 - 2*ATR) with daily EMA(50) trending down and volume > 1.5x 20-period average.
Exit when price returns to the EMA10 (middle band).
Keltner Channels adapt to volatility via ATR, making them effective in both trending and ranging markets.
The daily trend filter prevents counter-trend trades, improving win rate during market reversals.
Designed for 20-30 trades/year to minimize fee drag while capturing strong trends with institutional validation.
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
    
    # Calculate EMA(10) for Keltner middle band
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate ATR(10) for Keltner width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands
    keltner_upper = ema_10 + 2 * atr
    keltner_lower = ema_10 - 2 * atr
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(10, 20, 50), n):
        # Skip if data not available
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to EMA10 (middle band)
            if close[i] <= ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to EMA10 (middle band)
            if close[i] >= ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: price breaks above Keltner upper with daily uptrend
                if (close[i] > keltner_upper[i] and close[i-1] <= keltner_upper[i-1] and 
                    daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Keltner lower with daily downtrend
                elif (close[i] < keltner_lower[i] and close[i-1] >= keltner_lower[i-1] and 
                      daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals