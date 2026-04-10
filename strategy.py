#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h EMA200 trend filter
# - Long when price breaks above Donchian(20) high with volume > 1.5x 20-bar avg AND 12h close > EMA200
# - Short when price breaks below Donchian(20) low with volume > 1.5x 20-bar avg AND 12h close < EMA200
# - Exit when price crosses Donchian(20) midpoint (mean of 20-bar high/low)
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Targets ~30 trades/year (120 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves
# - Volume confirmation filters false breakouts
# - 12h EMA200 ensures alignment with higher timeframe trend

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period)
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Pre-compute 12h EMA(200) for trend filter
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high with volume spike and 12h uptrend
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_200_12h_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short signal: price breaks below Donchian low with volume spike and 12h downtrend
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_200_12h_aligned[i]):
                position = -1
                signals[i] = -0.30
        else:  # Have position - look for exit
            # Exit when price crosses Donchian midpoint
            if position == 1 and prices['close'].iloc[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals