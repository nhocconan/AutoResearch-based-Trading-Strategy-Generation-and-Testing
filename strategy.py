#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume spike confirmation.
# Long when price > Alligator Jaw (teeth) AND 1d EMA34 rising AND volume > 2x 20-bar average.
# Short when price < Alligator Jaw (teeth) AND 1d EMA34 falling AND volume > 2x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to capture medium-term trends.
# Williams Alligator (Smoothed MA crossover) identifies trend presence and direction.
# 1d EMA34 trend filter ensures alignment with higher timeframe momentum.
# Volume spike requirement reduces false signals and improves signal quality.
# Target: 50-150 total trades over 4 years (12-37/year) for BTC/ETH/SOL.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d EMA34 slope (rising/falling)
    ema_34_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_34_rising = ema_34_slope > 0
    ema_34_falling = ema_34_slope < 0
    
    # Williams Alligator on 6h data: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    def smma(arr, period):
        """Smoothed Moving Average - Williams Alligator uses this"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw - 13-period SMMA
    teeth = smma(close, 8)  # Teeth - 8-period SMMA
    lips = smma(close, 5)   # Lips - 5-period SMMA
    
    # Volume confirmation: current 6h volume > 2x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and EMA calculation
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Alligator signals: price relationship to Jaw (teeth)
        # In uptrend: Lips > Teeth > Jaw, price above Jaw
        # In downtrend: Lips < Teeth < Jaw, price below Jaw
        price_above_jaw = curr_close > jaw[i]
        price_below_jaw = curr_close < jaw[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price above Jaw AND 1d EMA34 rising AND volume confirmation
            if (price_above_jaw and 
                ema_34_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price below Jaw AND 1d EMA34 falling AND volume confirmation
            elif (price_below_jaw and 
                  ema_34_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Jaw OR 1d EMA34 falls (trend change)
            if (curr_close < jaw[i] or 
                ema_34_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Jaw OR 1d EMA34 rises (trend change)
            if (curr_close > jaw[i] or 
                ema_34_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals