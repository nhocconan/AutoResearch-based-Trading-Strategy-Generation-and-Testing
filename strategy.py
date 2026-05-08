#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_Alligator_1dTrend"
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
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_sum = np.zeros_like(tr)
        dm_plus_sum = np.zeros_like(dm_plus)
        dm_minus_sum = np.zeros_like(dm_minus)
        
        tr_sum[:period-1] = np.nan
        dm_plus_sum[:period-1] = np.nan
        dm_minus_sum[:period-1] = np.nan
        
        if len(tr) >= period:
            tr_sum[period-1] = np.nansum(tr[:period])
            dm_plus_sum[period-1] = np.nansum(dm_plus[:period])
            dm_minus_sum[period-1] = np.nansum(dm_minus[:period])
            
            for i in range(period, len(tr)):
                tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i]
                dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / period) + dm_plus[i]
                dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / period) + dm_minus[i]
        
        # DI values
        di_plus = np.full_like(tr, np.nan)
        di_minus = np.full_like(tr, np.nan)
        
        valid = tr_sum != 0
        di_plus[valid] = 100 * (dm_plus_sum[valid] / tr_sum[valid])
        di_minus[valid] = 100 * (dm_minus_sum[valid] / tr_sum[valid])
        
        # DX
        dx = np.full_like(tr, np.nan)
        di_sum = di_plus + di_minus
        valid_dx = di_sum != 0
        dx[valid_dx] = 100 * np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]
        
        # ADX
        adx = np.full_like(tr, np.nan)
        adx[:2*period-2] = np.nan
        
        if len(dx) >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Alligator lines (13,8,5 smoothed with 8,5,3)
    def smoothed_ma(data, period):
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        return pd.Series(sma).ewm(alpha=2/(period+1), adjust=False).mean().values
    
    jaw = smoothed_ma(close, 13)  # Smoothed with 8-period
    teeth = smoothed_ma(close, 8)  # Smoothed with 5-period
    lips = smoothed_ma(close, 5)   # Smoothed with 3-period
    
    # Alligator alignment: bullish when lips > teeth > jaw, bearish when lips < teeth < jaw
    alligator_bull = (lips > teeth) & (teeth > jaw)
    alligator_bear = (lips < teeth) & (teeth < jaw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14*2+2)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending) + Alligator bullish + price > 1d EMA50
            long_cond = (adx[i] > 25) and alligator_bull[i] and (close[i] > ema_50_1d_aligned[i])
            # Short: ADX > 25 (trending) + Alligator bearish + price < 1d EMA50
            short_cond = (adx[i] > 25) and alligator_bear[i] and (close[i] < ema_50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator bearish crossover (lips < teeth)
            if lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator bullish crossover (lips > teeth)
            if lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals