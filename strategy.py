#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(20) trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel AND 1w EMA(20) is rising (uptrend).
# Short when price breaks below lower Donchian channel AND 1w EMA(20) is falling (downtrend).
# Volume spike filter (1.5x 20-period average) reduces false breakouts.
# Target: 20-50 total trades over 4 years (5-12/year) to stay within fee limits.
# Designed to capture strong trends while avoiding false breakouts in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - avoid low-liquidity hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data for Donchian channels (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian(20) channels
    upper_donchian = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(20) for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or \
           np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Donchian + rising 1w EMA + volume spike
            if (close[i] > upper_donchian[i] and 
                ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian + falling 1w EMA + volume spike
            elif (close[i] < lower_donchian[i] and 
                  ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite breakout or EMA flattening
            if position == 1:
                if (close[i] < lower_donchian[i] or 
                    ema_20_1w_aligned[i] <= ema_20_1w_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > upper_donchian[i] or 
                    ema_20_1w_aligned[i] >= ema_20_1w_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA20_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0