# Hypothesis: 1d Donchian channel breakout with weekly EMA trend filter and weekly volume confirmation
# Donchian breakout (price > 20-period high or < 20-period low) captures breakouts in both bull and bear markets.
# Weekly EMA20 trend filter ensures we only trade in direction of higher timeframe trend.
# Weekly volume confirmation (> 1.5x 20-week average) avoids false breakouts.
# Weekly timeframe reduces noise and false signals, leading to fewer trades and lower fee drag.
# Designed for 1d timeframe targeting 10-25 trades per year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend and volume filters (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly volume 20-period average for spike detection
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    # Daily Donchian channels (20-period)
    # We need at least 20 periods of data, so we calculate directly
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after enough data for all indicators
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + weekly uptrend + weekly volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + weekly downtrend + weekly volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian band or trend reversal
            if position == 1:
                # Exit on return to Donchian low or trend reversal
                if (close[i] <= donchian_low[i] or 
                    close[i] < ema_20_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on return to Donchian high or trend reversal
                if (close[i] >= donchian_high[i] or 
                    close[i] > ema_20_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA20_1wVolSpike"
timeframe = "1d"
leverage = 1.0