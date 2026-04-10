#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w trend filter + volume confirmation
# - Long when price breaks above 20-day high AND 1w EMA21 > 1w EMA50 (uptrend) AND volume > 1.5x 20-day average
# - Short when price breaks below 20-day low AND 1w EMA21 < 1w EMA50 (downtrend) AND volume > 1.5x 20-day average
# - Exit when price crosses 10-day EMA in opposite direction
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - Donchian breakouts capture strong momentum moves
# - 1w EMA filter ensures we only trade with the higher timeframe trend
# - Volume confirmation ensures breakouts have conviction
# - Works in both bull and bear markets by following the 1w trend

name = "1d_1w_donchian_trend_volume_v1"
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
    
    # Pre-compute 1d Donchian channels (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d 10-day EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Pre-compute 1w EMA21 and EMA50 for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w trend: 1 = uptrend (EMA21 > EMA50), -1 = downtrend (EMA21 < EMA50), 0 = unclear
    ema_trend_1w = np.where(ema_21_1w > ema_50_1w, 1, np.where(ema_21_1w < ema_50_1w, -1, 0))
    
    # Align HTF indicators to 1d timeframe
    ema_trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_trend_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_10[i]) or 
            np.isnan(ema_trend_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Donchian breakout above 20-day high AND 1w uptrend AND volume spike
            if (close[i] > highest_high[i] and 
                ema_trend_1w_aligned[i] == 1 and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Donchian breakout below 20-day low AND 1w downtrend AND volume spike
            elif (close[i] < lowest_low[i] and 
                  ema_trend_1w_aligned[i] == -1 and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses 10-day EMA in opposite direction
            exit_long = (position == 1 and close[i] < ema_10[i])
            exit_short = (position == -1 and close[i] > ema_10[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals