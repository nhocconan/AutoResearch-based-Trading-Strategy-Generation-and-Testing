#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1w trend filter and volume confirmation.
# Long when: 6h Williams %R < -80 (oversold) AND 1w close > 1w EMA34 AND 6h volume > 1.5x 20-period average
# Short when: 6h Williams %R > -20 (overbought) AND 1w close < 1w EMA34 AND 6h volume > 1.5x 20-period average
# Uses Williams %R for mean reversion extremes, 1w EMA34 for major trend alignment, volume spike for conviction.
# Target: 12-37 trades/year on 6h. Discrete sizing 0.25 to balance return and fee drag.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) by fading extremes with aligned 1w trend.

name = "6h_WilliamsR_1wEMA34_VolumeConfirm_v1"
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
    
    # Load 6h data ONCE before loop for price action and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for Williams %R calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w Williams %R (14-period)
    highest_high_1w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    close_1w = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r_1w = np.where(
        (highest_high_1w - lowest_low_1w) != 0,
        ((highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w)) * -100,
        -50  # neutral when range is zero
    )
    
    # Align Williams %R to 6h primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 6h volume average (20-period) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for 1w EMA34 and Williams %R
    
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
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 1w trend filter
        uptrend_1w = curr_close > curr_ema_34
        downtrend_1w = curr_close < curr_ema_34
        
        # Williams %R extremes
        oversold = curr_williams_r < -80
        overbought = curr_williams_r > -20
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold AND 1w uptrend AND volume confirmation
            if (oversold and 
                uptrend_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND 1w downtrend AND volume confirmation
            elif (overbought and 
                  downtrend_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (mean reversion) OR 1w trend turns down
            if (curr_williams_r > -50 or 
                not uptrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (mean reversion) OR 1w trend turns up
            if (curr_williams_r < -50 or 
                not downtrend_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals