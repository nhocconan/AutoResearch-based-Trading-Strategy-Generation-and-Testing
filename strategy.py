#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R (14) mean reversion + 1d EMA34 trend filter + volume spike.
# Long when %R < -80 (oversold) AND 1d EMA34 rising AND volume > 2.0x 20-bar average.
# Short when %R > -20 (overbought) AND 1d EMA34 falling AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 4h timeframe to capture reversals in both bull and bear markets via 1d EMA34 trend filter.
# Williams %R identifies exhaustion points; trend filter ensures we trade with the higher timeframe momentum; volume spike confirms conviction.

name = "4h_WilliamsR14_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d EMA34 slope (rising/falling)
    ema_34_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_34_rising = ema_34_slope > 0
    ema_34_falling = ema_34_slope < 0
    
    # Williams %R (14-period) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 4h volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 4h timeframe
        hour = hours[i]
        
        if np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Williams %R signals
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: oversold AND 1d EMA34 rising AND volume confirmation
            if (oversold and 
                ema_34_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought AND 1d EMA34 falling AND volume confirmation
            elif (overbought and 
                  ema_34_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (mean reversion) OR 1d EMA34 falls (trend change)
            if (williams_r[i] > -50 or 
                ema_34_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (mean reversion) OR 1d EMA34 rises (trend change)
            if (williams_r[i] < -50 or 
                ema_34_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals