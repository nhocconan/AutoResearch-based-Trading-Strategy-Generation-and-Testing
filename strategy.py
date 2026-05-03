#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND close > 1w EMA50 AND volume > 1.5x 20-period volume MA.
# Short when price breaks below Donchian(20) low AND close < 1w EMA50 AND volume > 1.5x 20-period volume MA.
# Exit when price touches opposite Donchian boundary (mean reversion in ranging markets) OR EMA50 cross.
# Uses 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year) with strict entry conditions.
# Donchian channels capture breakouts, 1w EMA50 filters for higher-timeframe trend, volume confirms participation.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 1d volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        # 1w EMA50 trend condition
        above_ema = close[i] > ema_50_1w_aligned[i]
        below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 1w EMA50 AND volume spike
            if high[i] > donchian_high[i] and above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1w EMA50 AND volume spike
            elif low[i] < donchian_low[i] and below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches Donchian low OR closes below 1w EMA50
            if low[i] <= donchian_low[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches Donchian high OR closes above 1w EMA50
            if high[i] >= donchian_high[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals