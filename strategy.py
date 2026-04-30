#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. In ranging markets, mean reversion from
# extreme levels works well. 1w EMA34 provides higher-timeframe trend bias to avoid counter-trend
# trades. Volume confirmation ensures breakouts/mean reversions have participation. Discrete sizing
# 0.25 balances risk and minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_ME_1wEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d Williams %R(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Avoid division by zero
    williams_r[highest_high == lowest_low] = -50
    
    # Align 1d Williams %R to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1w EMA(34) for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # warmup for volume MA and 1w EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Mean reversion long: oversold AND above 1w EMA34 (bullish bias)
                if (curr_williams_r < -80 and 
                    curr_close > curr_ema_34_1w):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Mean reversion short: overbought AND below 1w EMA34 (bearish bias)
                elif (curr_williams_r > -20 and 
                      curr_close < curr_ema_34_1w):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns to neutral territory (-50)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns to neutral territory (-50)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals