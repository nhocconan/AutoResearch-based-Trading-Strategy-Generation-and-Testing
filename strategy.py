#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Percent Range (%R) with 1d trend filter and volume confirmation.
# Long when: %R < -80 (oversold) AND 1d close > 1d EMA50 AND 6h volume > 1.5x 20-period average
# Short when: %R > -20 (overbought) AND 1d close < 1d EMA50 AND 6h volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 12-37 trades/year on 6h.
# %R identifies mean reversion extremes, 1d EMA50 filters for higher timeframe trend alignment,
# volume spike confirms conviction. Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend).

name = "6h_WilliamsR_1dTrend_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for Williams %R
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams %R on 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_6h = pd.Series(df_6h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(df_6h['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - df_6h['close'].values) / (highest_high_6h - lowest_low_6h + 1e-10)
    
    # Align Williams %R to 6h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r_6h)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h volume average (20-period) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA and Williams %R
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema_50 = ema_50_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Williams %R conditions
        oversold = curr_williams_r < -80
        overbought = curr_williams_r > -20
        
        # 1d trend filter: price above/below EMA50
        uptrend_1d = curr_close > curr_ema_50
        downtrend_1d = curr_close < curr_ema_50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: oversold %R AND 1d uptrend AND volume confirmation
            if (oversold and 
                uptrend_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought %R AND 1d downtrend AND volume confirmation
            elif (overbought and 
                  downtrend_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: %R rises above -50 (exit oversold zone) OR 1d trend turns down
            if (curr_williams_r > -50 or 
                not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: %R falls below -50 (exit overbought zone) OR 1d trend turns up
            if (curr_williams_r < -50 or 
                not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals