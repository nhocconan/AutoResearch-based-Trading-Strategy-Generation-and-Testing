#!/usr/bin/env python3
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
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Keltner channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Keltner Channel on daily: EMA(20) +/- ATR(20)
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # EMA(20)
    ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # ATR(20)
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner bands
    upper = ema20 + 2.0 * atr20
    lower = ema20 - 2.0 * atr20
    
    # Align to 4h timeframe (shifted by 1 day for lookback)
    ema20_4h = align_htf_to_ltf(prices, df_1d, ema20)
    upper_4h = align_htf_to_ltf(prices, df_1d, upper)
    lower_4h = align_htf_to_ltf(prices, df_1d, lower)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any value is not ready
        if (np.isnan(ema20_4h[i]) or np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below EMA20 (trend reversal)
            if close[i] < ema20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA20 (trend reversal)
            if close[i] > ema20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume must be present for any entry
            if not volume_spike[i]:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above upper Keltner band with volume (bullish breakout)
            if close[i] > upper_4h[i] and close[i-1] <= upper_4h[i-1]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Keltner band with volume (bearish breakout)
            elif close[i] < lower_4h[i] and close[i-1] >= lower_4h[i-1]:
                position = -1
                signals[i] = -0.25
    
    return signals