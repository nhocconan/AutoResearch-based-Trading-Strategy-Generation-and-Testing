#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(40) breakout with 1w EMA34 trend filter and volume spike confirmation.
# Enter long when price breaks above 40-day high with 1w EMA34 uptrend and volume > 2x 20-bar average.
# Enter short when price breaks below 40-day low with 1w EMA34 downtrend and volume > 2x 20-bar average.
# Exit when price retraces to the 20-day EMA.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 40-100 total trades over 4 years (10-25/year).
# Longer Donchian period reduces false breakouts; volume spike ensures institutional participation;
# 1w EMA34 provides smooth higher timeframe trend filter. Designed to capture strong trends
# in both bull (breakouts) and bear (breakdowns) markets while avoiding choppy regimes.

name = "1d_Donchian40_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 1d
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate Donchian channels (40-period)
    # Highest high over past 40 days
    highest_40 = pd.Series(high).rolling(window=40, min_periods=40).max().values
    # Lowest low over past 40 days
    lowest_40 = pd.Series(low).rolling(window=40, min_periods=40).min().values
    
    # Volume confirmation: >2x 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    # Exit condition: 20-day EMA
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient history for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(highest_40[i]) or np.isnan(lowest_40[i]) or np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        vol_spike = volume_spike[i]
        
        # 1w EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > 40-day high, EMA34 up, volume spike
            if price > highest_40[i] and ema_trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price < 40-day low, EMA34 down, volume spike
            elif price < lowest_40[i] and ema_trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at 20-day EMA
            if price <= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at 20-day EMA
            if price >= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals