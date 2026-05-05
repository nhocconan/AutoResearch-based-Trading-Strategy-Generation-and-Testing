#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above Donchian upper channel AND price > EMA50(1w) AND volume > 2.0x 20-period average
# Short when price breaks below Donchian lower channel AND price < EMA50(1w) AND volume > 2.0x 20-period average
# Exit when price crosses back over Donchian middle (mean reversion) OR trend flips (price crosses EMA50(1w))
# Donchian(20) provides clear price structure with proven edge on SOLUSDT
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws in bear markets
# Volume spike confirms institutional participation
# Target: 7-25 trades/year per symbol (30-100 total over 4 years)
# Discrete sizing (0.25) to limit fee drag

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 1d
    if len(high) >= 20 and len(low) >= 20:
        upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
        middle_channel = (upper_channel + lower_channel) / 2
    else:
        upper_channel = np.full(n, np.nan)
        lower_channel = np.full(n, np.nan)
        middle_channel = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND price > EMA50(1w) AND volume spike
            if (close[i] > upper_channel[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND price < EMA50(1w) AND volume spike
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Donchian middle (mean reversion) OR price < EMA50(1w) (trend flip)
            if (close[i] < middle_channel[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Donchian middle (mean reversion) OR price > EMA50(1w) (trend flip)
            if (close[i] > middle_channel[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals