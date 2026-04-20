# 12h_1d_Triple_Pullback_Strategy
# 12h timeframe with 1d HTF
# Strategy: Enter on pullbacks to EMA(34) in trending markets (ADX > 25) with volume confirmation
# Exit when price closes opposite EMA or trend weakens
# Works in bull/bear: trend following with pullback entries captures momentum while avoiding whipsaws
# Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25

name = "12h_1d_Triple_Pullback_Strategy"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for trend and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smoothed_avg(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.nansum(x[:period])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr = smoothed_avg(tr, 14)
    dm_plus_smooth = smoothed_avg(dm_plus, 14)
    dm_minus_smooth = smoothed_avg(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_avg(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h EMA(34) for entry timing
    close_12h = prices['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume average for spike detection
    volume_12h = prices['volume'].values
    vol_avg_12h = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i >= 19:  # 20-period average
            vol_avg_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_34_1d_aligned[i]
        adx_val = adx_aligned[i]
        ema_fast = ema_34_12h[i]
        vol_avg = vol_avg_12h[i]
        
        # Skip if any values are NaN
        if np.isnan(ema_trend) or np.isnan(adx_val) or np.isnan(ema_fast) or np.isnan(vol_avg):
            continue
            
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x average
        vol_confirm = current_volume > 1.3 * vol_avg
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: price pulls back to EMA in uptrend with volume
            if (strong_trend and 
                current_close > ema_trend and  # Above 1d EMA (uptrend)
                current_close <= ema_fast * 1.02 and  # Near 12h EMA (pullback)
                current_close >= ema_fast * 0.98 and
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to EMA in downtrend with volume
            elif (strong_trend and 
                  current_close < ema_trend and  # Below 1d EMA (downtrend)
                  current_close >= ema_fast * 0.98 and  # Near 12h EMA (pullback)
                  current_close <= ema_fast * 1.02 and
                  vol_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakens or price crosses below EMA
            if (adx_val < 20 or  # Trend weakening
                current_close < ema_trend):  # Below 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakens or price crosses above EMA
            if (adx_val < 20 or  # Trend weakening
                current_close > ema_trend):  # Above 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals