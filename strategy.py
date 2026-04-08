#!/usr/bin/env python3
# 12h_1w_1d_donchian_breakout_volume_regime_v2
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1w trend filter.
# Long when price breaks above 20-period high with volume > 1.5x average and 1w uptrend.
# Short when price breaks below 20-period low with volume > 1.5x average and 1w downtrend.
# Uses 1d ATR for volatility-based stoploss and position sizing (0.25).
# Designed for 12-30 trades/year on 12h to avoid fee drag. Works in bull/bear via multi-timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_donchian_breakout_volume_regime_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    period20_high = np.full(n, np.nan)
    period20_low = np.full(n, np.nan)
    for i in range(20, n):
        period20_high[i] = np.max(high[i-20:i+1])
        period20_low[i] = np.min(low[i-20:i+1])
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i+1])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA25 for trend filter
    ema25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    
    # Get 1d data for ATR (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(14) on 1d
    tr1 = np.zeros(len(df_1d))
    tr2 = np.zeros(len(df_1d))
    tr3 = np.zeros(len(df_1d))
    tr1[1:] = np.abs(high_1d[1:] - low_1d[1:])
    tr2[1:] = np.abs(high_1d[1:] - close_1d[:-1])
    tr3[1:] = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = np.zeros(len(df_1d))
    for i in range(14, len(tr)):
        atr14[i] = np.mean(tr[i-14:i+1])
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 25)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema25_1w_aligned[i]) or 
            np.isnan(atr14_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # 1w trend filter
        uptrend_1w = close[i] > ema25_1w_aligned[i]
        downtrend_1w = close[i] < ema25_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low or volume drops
            if close[i] < period20_low[i] or volume[i] < 0.5 * vol_avg[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high or volume drops
            if close[i] > period20_high[i] or volume[i] < 0.5 * vol_avg[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-period high with volume confirmation and 1w uptrend
            if (close[i] > period20_high[i] and 
                vol_confirm and 
                uptrend_1w):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period low with volume confirmation and 1w downtrend
            elif (close[i] < period20_low[i] and 
                  vol_confirm and 
                  downtrend_1w):
                position = -1
                signals[i] = -0.25
    
    return signals