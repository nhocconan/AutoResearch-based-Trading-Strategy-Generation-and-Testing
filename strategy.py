#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels work in both bull and bear markets.
# Long when Williams %R crosses above -80 from below AND price > 1d EMA50 AND volume > 1.8x 20-bar average.
# Short when Williams %R crosses below -20 from above AND price < 1d EMA50 AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25. Session filter 08-20 UTC to avoid low-liquidity hours.
# Volume threshold increased to 1.8x to reduce trade frequency and avoid overtrading.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_WilliamsR_1dEMA50_Trend_VolumeSpike_v1"
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
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 12h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R, EMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # Williams %R reversal signals
        # Long: Williams %R crosses above -80 from below (oversold reversal)
        williams_r_long_signal = (williams_r[i-1] < -80) and (williams_r[i] >= -80)
        # Short: Williams %R crosses below -20 from above (overbought reversal)
        williams_r_short_signal = (williams_r[i-1] > -20) and (williams_r[i] <= -20)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R reversal above -80 AND price > 1d EMA50 AND volume confirmation
            if (williams_r_long_signal and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R reversal below -20 AND price < 1d EMA50 AND volume confirmation
            elif (williams_r_short_signal and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR price < 1d EMA50 (trend change)
            if (williams_r[i] >= -20 or 
                not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR price > 1d EMA50 (trend change)
            if (williams_r[i] <= -80 or 
                not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals