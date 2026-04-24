#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price breaks above Camarilla R3 AND 1d EMA34 rising (bullish trend) AND volume > 1.5x 20-period average
- Short when price breaks below Camarilla S3 AND 1d EMA34 falling (bearish trend) AND volume > 1.5x 20-period average
- Exit on opposite Camarilla break (R3/S3) or trend reversal (EMA34 direction change)
- Position size fixed at 0.25 to balance risk and reward, minimizing fee churn
- Uses 4h primary with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Camarilla levels provide precise intraday support/resistance; EMA34 filter avoids chop and confirms trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla R3, S3 from previous 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # EMA34 direction: rising if current > previous, falling if current < previous
    ema_34_rising = ema_34_1d_aligned > np.roll(ema_34_1d_aligned, 1)
    ema_34_falling = ema_34_1d_aligned < np.roll(ema_34_1d_aligned, 1)
    # Handle first bar
    ema_34_rising[0] = False
    ema_34_falling[0] = False
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 AND rising EMA34 AND volume confirmed
            if close[i] > camarilla_r3_aligned[i] and ema_34_rising[i] and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 AND falling EMA34 AND volume confirmed
            elif close[i] < camarilla_s3_aligned[i] and ema_34_falling[i] and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below S3 OR EMA34 starts falling
            if close[i] < camarilla_s3_aligned[i] or ema_34_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R3 OR EMA34 starts rising
            if close[i] > camarilla_r3_aligned[i] or ema_34_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0