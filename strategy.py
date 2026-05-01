#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R reversal with 4h EMA50 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold bounce) AND 4h EMA50 rising AND volume > 1.3x 20-bar average.
# Short when Williams %R crosses below -20 (overbought rejection) AND 4h EMA50 falling AND volume > 1.3x 20-bar average.
# Uses discrete sizing 0.20 to minimize fee churn. Designed for 1h timeframe to capture mean-reversion swings within the trend.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) via 4h EMA50 slope filter.

name = "1h_WilliamsR_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 calculation
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # 4h EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Williams %R cross signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]
    williams_r_cross_above_80 = (williams_r_prev <= -80) & (williams_r > -80)
    williams_r_cross_below_20 = (williams_r_prev >= -20) & (williams_r < -20)
    
    # Volume confirmation: current 1h volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = volume[i] > (vol_ma[i] * 1.3)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND 4h EMA50 rising AND volume confirmation
            if (williams_r_cross_above_80[i] and 
                ema_50_rising[i] and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: Williams %R crosses below -20 AND 4h EMA50 falling AND volume confirmation
            elif (williams_r_cross_below_20[i] and 
                  ema_50_falling[i] and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR 4h EMA50 falls (trend change)
            if (williams_r[i] < -50 and williams_r_prev[i] >= -50) or ema_50_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR 4h EMA50 rises (trend change)
            if (williams_r[i] > -50 and williams_r_prev[i] <= -50) or ema_50_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals