#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX regime filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; ADX filters for trending vs ranging markets
# Long: %R < -80 (oversold) + ADX > 25 (trending) + price > 1d EMA(50) + volume spike
# Short: %R > -20 (overbought) + ADX > 25 (trending) + price < 1d EMA(50) + volume spike
# Uses discrete position sizing (0.25) to minimize fee churn
# Targets 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits for 6h timeframe

name = "6h_WilliamsR_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        plus_dm = np.zeros_like(high_arr)
        minus_dm = np.zeros_like(low_arr)
        tr = np.zeros_like(high_arr)
        
        for i in range(1, len(high_arr)):
            up_move = high_arr[i] - high_arr[i-1]
            down_move = low_arr[i-1] - low_arr[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nansum(tr[1:period+1]) if not np.any(np.isnan(tr[1:period+1])) else np.nan
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high_arr)
        minus_di = np.zeros_like(low_arr)
        dx = np.zeros_like(high_arr)
        
        for i in range(period, len(tr)):
            if atr[i] != 0:
                plus_di[i] = (np.nansum(plus_dm[i-period+1:i+1]) / atr[i]) * 100
                minus_di[i] = (np.nansum(minus_dm[i-period+1:i+1]) / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        adx = np.full_like(dx, np.nan)
        if len(dx) >= 2*period:
            adx[2*period-1] = np.nanmean(dx[period:2*period]) if not np.any(np.isnan(dx[period:2*period])) else np.nan
            for i in range(2*period, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ADX(14)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF indicators to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Williams %R(14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R, volume MA, and HTF indicators)
    start_idx = 50  # buffer for 20-period volume MA and 14-period Williams %R
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) + ADX > 25 (trending) + price > 1d EMA + volume spike
            if williams_r[i] < -80 and adx_1d_aligned[i] > 25 and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + ADX > 25 (trending) + price < 1d EMA + volume spike
            elif williams_r[i] > -20 and adx_1d_aligned[i] > 25 and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (momentum fading) or ADX < 20 (losing trend) or price < 1d EMA
            if williams_r[i] > -50 or adx_1d_aligned[i] < 20 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (momentum fading) or ADX < 20 (losing trend) or price > 1d EMA
            if williams_r[i] < -50 or adx_1d_aligned[i] < 20 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals