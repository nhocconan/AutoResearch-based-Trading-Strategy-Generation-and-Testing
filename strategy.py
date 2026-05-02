#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
# Donchian channels identify key support/resistance based on 20-day high/low; breakouts above
# upper band or below lower band with volume confirmation indicate strong momentum. 1w EMA200
# ensures trades align with long-term trend to avoid false breakouts in choppy markets. Designed
# for 30-100 total trades over 4 years (7-25/year) on 1d timeframe. Works in bull markets
# (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend) by only
# taking trades in direction of 1w EMA200.

name = "1d_Donchian20_1wEMA200_Volume"
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
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 20-period Donchian channels (using 1d data)
    # Upper band = 20-period high, Lower band = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period average (20*1d = 20 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and EMA200)
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian band with volume spike AND price > 1w EMA200 (bullish trend)
            if (close[i] > donchian_upper[i] and 
                volume_spike[i] and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian band with volume spike AND price < 1w EMA200 (bearish trend)
            elif (close[i] < donchian_lower[i] and 
                  volume_spike[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below upper Donchian band (failed breakout) OR price below 1w EMA200 (trend change)
            if close[i] < donchian_upper[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above lower Donchian band (failed breakdown) OR price above 1w EMA200 (trend change)
            if close[i] > donchian_lower[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals