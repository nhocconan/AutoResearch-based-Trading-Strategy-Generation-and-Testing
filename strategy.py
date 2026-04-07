#!/usr/bin/env python3
"""
1d_price_channel_1w_trend_volume_v1
Hypothesis: On 1d timeframe, use weekly price channels (Donchian/ATR-based) with volume confirmation.
- Long when price breaks above upper channel (20-day high + ATR) with volume > 1.5x avg and weekly trend up
- Short when price breaks below lower channel (20-day low - ATR) with volume > 1.5x avg and weekly trend down
- Exit on opposite channel touch or trend reversal
Uses weekly trend filter to avoid counter-trend trades in both bull and bear markets.
Target: 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_price_channel_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50w = df_1w['close'].ewm(span=50, adjust=False).mean()
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w.values)
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily ATR(14) for channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Dynamic channels: Donchian ± ATR
    upper_channel = donchian_high + atr
    lower_channel = donchian_low - atr
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50w_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches lower channel or weekly trend turns bearish
            if close[i] <= lower_channel[i] or close[i] < ema_50w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches upper channel or weekly trend turns bullish
            if close[i] >= upper_channel[i] or close[i] > ema_50w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper channel with volume in uptrend
            if (close[i] > upper_channel[i] and
                vol_confirm and 
                close[i] > ema_50w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel with volume in downtrend
            elif (close[i] < lower_channel[i] and
                  vol_confirm and 
                  close[i] < ema_50w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals