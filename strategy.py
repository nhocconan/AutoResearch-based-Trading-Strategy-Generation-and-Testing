#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-day Donchian high with volume > 1.5x 20-day avg AND 1w close > 1w EMA50
# - Short when price breaks below 20-day Donchian low with volume > 1.5x 20-day avg AND 1w close < 1w EMA50
# - Exit when price returns to 20-day Donchian midpoint
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15 trades/year (60 total over 4 years) to avoid fee drag
# - Donchian breakouts work well in both bull (trend continuation) and bear (mean reversion after overextension)
# - Volume confirmation filters false breakouts
# - 1w EMA50 trend filter ensures alignment with higher timeframe

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d Donchian(20) channels
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    midpoint_20 = (high_20 + low_20) / 2.0
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(midpoint_20[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above 20-day high with volume spike and 1w uptrend
            if (prices['high'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below 20-day low with volume spike and 1w downtrend
            elif (prices['low'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to 20-day Donchian midpoint
            if position == 1 and prices['close'].iloc[i] < midpoint_20[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > midpoint_20[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals