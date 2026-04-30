#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper AND price > 1w EMA34 AND volume > 2.0x 20-bar average.
# Short when price breaks below Donchian lower AND price < 1w EMA34 AND volume > 2.0x 20-bar average.
# Exit when price crosses Donchian midline. Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull/bear via 1w EMA34 trend filter and volume confirmation.

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels on 1d data (use lookback of 20 periods)
    # Upper channel: highest high of last 20 bars (excluding current)
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    # Lower channel: lowest low of last 20 bars (excluding current)
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    # Midline: average of upper and lower
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian upper, uptrend (price > 1w EMA34), volume confirmation
            if (curr_high > donchian_upper[i] and 
                curr_close > ema_34_1w_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Donchian lower, downtrend (price < 1w EMA34), volume confirmation
            elif (curr_low < donchian_lower[i] and 
                  curr_close < ema_34_1w_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below Donchian midline
            if curr_close < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above Donchian midline
            if curr_close > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals