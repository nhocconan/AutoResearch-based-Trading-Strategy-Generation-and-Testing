#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 12h EMA50 trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 from below AND price > 12h EMA50 AND volume > 2.0x 20-bar average.
# Short when Williams %R crosses below -20 from above AND price < 12h EMA50 AND volume > 2.0x 20-bar average.
# Williams %R identifies overextended reversals, effective in both bull and bear markets.
# 12h EMA50 provides smooth trend alignment less prone to whipsaw than shorter EMAs.
# Volume spike (2.0x average) ensures only high-momentum reversals are traded.
# Session filter 08-20 UTC avoids low-liquidity hours. Discrete sizing 0.25 controls drawdown.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_WilliamsR_12hEMA50_Trend_VolumeSpike_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 calculation
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 12h trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R crossover signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]  # first value unchanged
    williams_r_cross_up = (williams_r > -80) & (williams_r_prev <= -80)  # cross above -80
    williams_r_cross_down = (williams_r < -20) & (williams_r_prev >= -20)  # cross below -20
    
    # Volume confirmation: current 4h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and Williams %R
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND price > 12h EMA50 AND volume confirmation
            if (williams_r_cross_up[i] and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < 12h EMA50 AND volume confirmation
            elif (williams_r_cross_down[i] and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (mean reversion) OR price < 12h EMA50 (trend change)
            if (williams_r[i] < -50 and williams_r_prev[i] >= -50) or \
               not price_above_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (mean reversion) OR price > 12h EMA50 (trend change)
            if (williams_r[i] > -50 and williams_r_prev[i] <= -50) or \
               not price_below_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals