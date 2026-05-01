#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Long when Alligator jaws (13-period smoothed median) > teeth (8-period smoothed median) > lips (5-period smoothed median)
# AND 1d EMA50 rising AND volume > 1.5x 20-bar average.
# Short when jaws < teeth < lips AND 1d EMA50 falling AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 12h timeframe to capture medium-term trends
# with fewer trades to avoid fee drag. Williams Alligator identifies trending vs ranging markets via three
# smoothed moving averages (jaws, teeth, lips) that diverge in trends and converge in ranges.
# 1d EMA50 ensures alignment with higher timeframe momentum. Volume confirmation reduces false signals.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d EMA50 slope (rising/falling)
    ema_50_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    ema_50_rising = ema_50_slope > 0
    ema_50_falling = ema_50_slope < 0
    
    # Williams Alligator: three smoothed moving averages of median price
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaws: 13-period SMMA, shifted 8 bars
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws, 8)
    jaws[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Volume confirmation: current 12h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and volume MA calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 12h timeframe
        hour = hours[i]
        
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Williams Alligator signals
        # Long when jaws > teeth > lips (diverged upward = uptrend)
        alligator_long = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])
        # Short when jaws < teeth < lips (diverged downward = downtrend)
        alligator_short = (jaws[i] < teeth[i]) and (teeth[i] < lips[i])
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend AND 1d EMA50 rising AND volume confirmation
            if (alligator_long and 
                ema_50_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend AND 1d EMA50 falling AND volume confirmation
            elif (alligator_short and 
                  ema_50_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator convergence (jaws < teeth OR teeth < lips) OR 1d EMA50 falls (trend change)
            if ((jaws[i] < teeth[i]) or 
                (teeth[i] < lips[i]) or 
                ema_50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator convergence (jaws > teeth OR teeth > lips) OR 1d EMA50 rises (trend change)
            if ((jaws[i] > teeth[i]) or 
                (teeth[i] > lips[i]) or 
                ema_50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals