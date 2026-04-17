#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
Long when price breaks above Donchian upper AND 12h EMA34 rising AND volume > 1.5x average.
Short when price breaks below Donchian lower AND 12h EMA34 falling AND volume > 1.5x average.
Exit when price touches opposite Donchian band or volume drops below average.
Uses 4h for entry/exit, 12h for trend filter. Donchian captures breakouts, volume confirms strength,
12h EMA34 filters for higher-timeframe trend alignment to reduce false signals.
Target: 75-200 total trades over 4 years (19-50/year) on BTC/ETH/SOL.
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
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h average volume (20-period) for confirmation
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(avg_volume[i]) or np.isnan(ema34_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        avg_vol = avg_volume[i]
        ema34 = ema34_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band AND volume > 1.5x average AND 12h EMA34 rising
            if price > upper and vol > 1.5 * avg_vol and ema34 > ema34_12h_aligned[max(i-1, start_idx)]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND volume > 1.5x average AND 12h EMA34 falling
            elif price < lower and vol > 1.5 * avg_vol and ema34 < ema34_12h_aligned[max(i-1, start_idx)]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches lower band OR volume drops below average
            if price < lower or vol < avg_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches upper band OR volume drops below average
            if price > upper or vol < avg_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeConfirm_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0