#!/usr/bin/env python3
# 6h_adx_dmi_volume_v1
# Hypothesis: 6h strategy using ADX(14) and DI crossover for trend strength/direction, filtered by volume confirmation.
# Long when +DI crosses above -DI, ADX > 25 (trending), and volume > 1.3x 20-period average.
# Short when -DI crosses above +DI, ADX > 25, and volume > 1.3x average.
# Uses discrete position sizing (±0.25) to minimize fee churn.
# Works in bull/bear via ADX trend filter and volume confirmation to avoid whipsaws.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_dmi_volume_v1"
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
    
    # Calculate ADX and DI (14-period) on primary timeframe
    # +DI = 100 * EMA(smoothed +DM) / ATR
    # -DI = 100 * EMA(smoothed -DM) / ATR
    # ADX = 100 * EMA(|+DI - -DI| / (+DI + -DI))
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing: alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize arrays
    atr = np.full(n, np.nan)
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    
    # First value: simple average
    if n >= period:
        atr[period-1] = np.nanmean(tr[:period])
        plus_dm_smooth[period-1] = np.nanmean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.nanmean(minus_dm[:period])
        
        # Wilder's smoothing for subsequent values
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Calculate +DI and -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    valid = ~np.isnan(atr) & (atr != 0)
    plus_di[valid] = 100 * plus_dm_smooth[valid] / atr[valid]
    minus_di[valid] = 100 * minus_dm_smooth[valid] / atr[valid]
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx[di_sum != 0] = 100 * di_diff[di_sum != 0] / di_sum[di_sum != 0]
    
    # ADX = EMA of DX
    adx = np.full(n, np.nan)
    if n >= period:
        adx[period-1] = np.nanmean(dx[:period])
        for i in range(period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Volume confirmation (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.3)  # Volume at least 1.3x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: -DI crosses above +DI (trend weakening/reversal)
            if minus_di[i] > plus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: +DI crosses above -DI (trend weakening/reversal)
            if plus_di[i] > minus_di[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: +DI crosses above -DI, ADX > 25 (strong trend), volume spike
            if (i > 0 and plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1] and
                adx[i] > 25 and vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: -DI crosses above +DI, ADX > 25 (strong trend), volume spike
            elif (i > 0 and minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1] and
                  adx[i] > 25 and vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals