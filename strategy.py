#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d trend filter (EMA50)
# - Long when price breaks above Donchian(20) high with volume > 1.5x 20-bar avg AND 1d close > 1d EMA50
# - Short when price breaks below Donchian(20) low with volume > 1.5x 20-bar avg AND 1d close < 1d EMA50
# - Exit when price returns to Donchian(20) midpoint
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - Donchian breakouts work well in trending markets; volume confirmation filters false breakouts
# - 1d trend filter ensures alignment with higher timeframe, reducing whipsaws in bear markets

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high with volume spike and 1d uptrend
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low with volume spike and 1d downtrend
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to Donchian midpoint
            if position == 1 and prices['close'].iloc[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals