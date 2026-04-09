#!/usr/bin/env python3
# 1d_donchian_breakout_weekly_volume_v1
# Hypothesis: Daily Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long: Close > upper Donchian(20) AND close > 1w EMA50 AND volume > 1.5x 20-day avg volume.
# Short: Close < lower Donchian(20) AND close < 1w EMA50 AND volume > 1.5x 20-day avg volume.
# Exit: Opposite Donchian breakout OR volume divergence (price moves against breakout but volume drops).
# Uses 1w EMA50 for higher timeframe trend filter to avoid counter-trend trades in bear markets.
# Volume confirmation filters weak breakouts. Target: 7-25 trades/year (30-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) - daily
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper_donchian = high_s.rolling(window=20, min_periods=20).max().values
    lower_donchian = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA50 for trend filter (weekly)
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or
            np.isnan(volume_ma[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below lower Donchian OR volume divergence (price up but volume down)
            if close[i] < lower_donchian[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian OR volume divergence (price down but volume down)
            if close[i] > upper_donchian[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above upper Donchian, above 1w EMA50, volume confirmed
            if (close[i] > upper_donchian[i] and close[i] > ema50_1w_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower Donchian, below 1w EMA50, volume confirmed
            elif (close[i] < lower_donchian[i] and close[i] < ema50_1w_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals