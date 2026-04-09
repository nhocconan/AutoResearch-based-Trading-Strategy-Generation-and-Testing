#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# - Uses 1w EMA(21) for weekly trend direction (long only in uptrend)
# - Enters long on break above 20-day Donchian high, exits on return to 10-day EMA
# - Volume confirmation: current volume > 1.5x 20-day average volume
# - Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years) to minimize fee drag
# - Breakout strategies work in both bull (continuation) and bear (sharp rallies) markets

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Pre-compute volume average (20-day)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 10-day EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_10[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: return to 10-day EMA
            if close[i] <= ema_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # Flat
            # Enter long: Donchian breakout with volume confirmation and 1w uptrend
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_ma_20[i] and 
                close[i] > ema_21_1w_aligned[i]):  # Price above 1w EMA (uptrend filter)
                position = 1
                signals[i] = 0.25
    
    return signals