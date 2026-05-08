# 6h_AtrBreakout_1dTrend_VolumeFilter
# Hypothesis: 6-hour ATR breakouts aligned with daily trend and volume spikes capture strong momentum moves
# in both bull and bear markets. The strategy uses a volatility-based breakout (ATR-based channel) to
# enter during expansion phases, with trend filter ensuring directionality and volume confirmation
# reducing false breakouts. Targets 50-150 total trades over 4 years to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_AtrBreakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility-based breakout channels
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.zeros(len(close_1d))
    atr_1d[0] = np.nan
    if len(tr) > 0:
        atr_1d[1] = np.mean(tr[:1]) if len(tr) >= 1 else np.nan
        for i in range(2, len(tr)+1):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i-1]) / 14
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6-period high/low for breakout channels (using 6h data)
    high_max6 = pd.Series(high).rolling(window=6, min_periods=6).max().values
    low_min6 = pd.Series(low).rolling(window=6, min_periods=6).min().values
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(high_max6[i]) or np.isnan(low_min6[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate dynamic breakout levels using daily ATR
        upper_break = high_max6[i] + (0.5 * atr_1d_aligned[i])
        lower_break = low_min6[i] - (0.5 * atr_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above upper channel with daily uptrend + volume spike
            long_cond = (close[i] > upper_break and 
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below lower channel with daily downtrend + volume spike
            short_cond = (close[i] < lower_break and 
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower channel (mean reversion)
            if close[i] < lower_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper channel (mean reversion)
            if close[i] > upper_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals