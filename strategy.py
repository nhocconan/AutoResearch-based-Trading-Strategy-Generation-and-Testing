#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (>1.5x 20-bar avg volume).
# Uses Donchian channel from 1d timeframe for breakout structure, 1w EMA50 for higher timeframe trend alignment, and volume spike for participation confirmation.
# Designed for BTC/ETH with discrete sizing (0.25) to minimize fee churn while capturing strong momentum moves in both bull and bear markets.
# Target: 30-100 total trades over 4 years on 1d timeframe.

name = "1d_Donchian20_1wEMA50_Volume_Breakout_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (based on prior 20 1d bars)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (already aligned via get_htf_data + shift(1))
    # No additional alignment needed as we used shift(1) on HTF data
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 20), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band, close > 1w EMA50, volume spike
            if (high[i] > donchian_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25  # Full position on breakout
                position = 1
            # SHORT: Price breaks below Donchian lower band, close < 1w EMA50, volume spike
            elif (low[i] < donchian_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25  # Full position on breakout
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # CONTINUE LONG: Reduce to half position if still above upper band and volume OK
            if (high[i] > donchian_high[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = 0.125  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks below upper band or low volume
                position = 0
        elif position == -1:
            # CONTINUE SHORT: Reduce to half position if still below lower band and volume OK
            if (low[i] < donchian_low[i] and 
                volume[i] > avg_volume[i]):
                signals[i] = -0.125  # Half position to reduce churn
            else:
                signals[i] = 0.0  # Exit if breaks above lower band or low volume
                position = 0
    
    return signals