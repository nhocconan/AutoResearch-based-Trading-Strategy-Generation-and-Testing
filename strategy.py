#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (EMA34 > EMA89) and volume confirmation
# - Uses 1w HTF for trend direction (EMA34 > EMA89 = uptrend, < = downtrend)
# - Long on break above 20-day high in uptrend, short on break below 20-day low in downtrend
# - Volume confirmation: current 1d volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 7-25 trades/year on 1d timeframe (28-100 total over 4 years)

name = "1d_1w_donchian_breakout_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 89:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMAs for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = pd.Series(close_1w).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMAs to 1d timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_89_1w_aligned[i]) or
            np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: 1w EMA34 > EMA89 = uptrend, < = downtrend
        uptrend = ema_34_1w_aligned[i] > ema_89_1w_aligned[i]
        downtrend = ema_34_1w_aligned[i] < ema_89_1w_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price closes below 20-day low
            if close[i] < low_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price closes above 20-day high
            if close[i] > high_ma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Breakout entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: break above 20-day high in uptrend
                if uptrend and close[i] > high_ma_20[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below 20-day low in downtrend
                elif downtrend and close[i] < low_ma_20[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals