#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses weekly EMA50 for major trend filter and 12h Donchian channels for breakout entries.
# Long when price breaks above 20-period Donchian high with volume > 1.8x 20-period MA and close > 1w EMA50 (bull regime).
# Short when price breaks below 20-period Donchian low with volume spike and close < 1w EMA50 (bear regime).
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Weekly EMA50 filters out counter-trend trades during major reversals (e.g., 2022 crash).
# Volume confirmation reduces false breakouts. Works in bull/bear via trend alignment.

name = "12h_Donchian20_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian channels (20-period)
    # We need to calculate this on 12h data directly since it's our primary timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 12h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above Donchian high with volume spike in bull regime
            if close_val > upper_channel and vol_spike and is_bull_regime:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: break below Donchian low with volume spike in bear regime
            elif close_val < lower_channel and vol_spike and is_bear_regime:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below Donchian low OR regime turns bear
            if close_val < lower_channel or not is_bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR regime turns bull
            if close_val > upper_channel or not is_bear_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals