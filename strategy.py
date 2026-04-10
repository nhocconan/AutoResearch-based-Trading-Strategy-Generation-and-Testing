#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA trend filter + volume confirmation
# - Long when price breaks above Donchian(20) high AND price > 1w EMA(50) AND volume > 1.5x 20-day average
# - Short when price breaks below Donchian(20) low AND price < 1w EMA(50) AND volume > 1.5x 20-day average
# - Exit when price crosses Donchian(10) midpoint (mean reversion) OR opposite breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Donchian breakouts capture strong trends in both bull and bear markets
# - 1w EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation ensures breakouts have conviction and reduces false signals

name = "1d_1w_donchian_volume_trend_v1"
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
    
    # Pre-compute 1d Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) for breakout signals
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exit signals (midpoint)
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (highest_high_10 + lowest_low_10) / 2
    
    # Pre-compute 1d volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(highest_high_10[i]) or np.isnan(lowest_low_10[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Donchian(20) breakout AND price > 1w EMA(50) AND volume spike
            if (close[i] > highest_high_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Donchian(20) breakdown AND price < 1w EMA(50) AND volume spike
            elif (close[i] < lowest_low_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian(10) midpoint OR opposite breakout with volume
            exit_long = (position == 1 and 
                        (close[i] < donchian_mid_10[i] or
                         (close[i] < lowest_low_20[i] and volume_spike[i])))
            exit_short = (position == -1 and 
                         (close[i] > donchian_mid_10[i] or
                          (close[i] > highest_high_20[i] and volume_spike[i])))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals