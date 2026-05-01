#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions. Long when %R crosses above -80 from below (oversold bounce),
# short when %R crosses below -20 from above (overbought reversal). Uses 1d EMA50 for trend alignment that works
# in both bull and bear markets. Volume confirmation requires 2.0x 20-bar average to ensure momentum behind moves.
# Discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_WilliamsR_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, -100 * (highest_high - close) / rr, -50.0)
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_r_long = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_r_short = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    # Handle first element
    williams_r_long[0] = False
    williams_r_short[0] = False
    
    # Volume confirmation: current 4h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA50 AND volume confirmation
            if (williams_r_long[i] and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) AND price < 1d EMA50 AND volume confirmation
            elif (williams_r_short[i] and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR price < 1d EMA50 (trend change)
            if (williams_r_short[i] or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR price > 1d EMA50 (trend change)
            if (williams_r_long[i] or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals