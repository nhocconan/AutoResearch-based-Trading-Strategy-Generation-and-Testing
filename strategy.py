#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter. Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13.
# In bull regimes (1d ADX > 25), go long when Bear Power turns up from below zero (bullish divergence).
# In bear regimes (1d ADX > 25), go short when Bull Power turns down from above zero (bearish divergence).
# In ranging regimes (1d ADX <= 25), fade extremes: long when Bull Power crosses below -0.5*ATR and turns up,
# short when Bull Power crosses above 0.5*ATR and turns down.
# Uses volume confirmation to avoid false signals. Discrete sizing 0.25 to manage drawdown.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_ElderRay_1dADX_Regime_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for ADX and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d ADX calculation (using Welles Wilder's method)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (using Wilder's smoothing: alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # first value is simple average
            result[period-1] = np.nanmean(data[1:period]) if np.any(~np.isnan(data[1:period])) else 0
            # subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
                else:
                    result[i] = np.nan
        return result
    
    atr_1d = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, 14)
    
    # Align 1d indicators to 6h
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    # 6h ATR for dynamic thresholds
    def calculate_atr(high, low, close, period):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = WilderSmooth(tr, period)
        return atr
    
    atr_6h = calculate_atr(high, low, close, 14)
    
    # Volume confirmation: 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Check for valid data
        if (np.isnan(ema_13_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Regime determination from 1d ADX
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] <= 25
        
        if position == 0:  # Flat - look for new entries
            if is_trending:
                # In trending markets, trade with the trend using Elder Ray divergence
                # Long: Bear Power turning up from below zero (bullish divergence in downtrend)
                # Short: Bull Power turning down from above zero (bearish divergence in uptrend)
                if (i >= 2 and 
                    bear_power[i-2] < 0 and bear_power[i-1] < 0 and bear_power[i] > bear_power[i-1] and  # Bear Power turning up
                    volume_confirm):
                    signals[i] = 0.25
                    position = 1
                elif (i >= 2 and 
                      bull_power[i-2] > 0 and bull_power[i-1] > 0 and bull_power[i] < bull_power[i-1] and  # Bull Power turning down
                      volume_confirm):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # ranging market
                # Fade extremes: look for reversals at extended levels
                # Long: Bull Power crosses below -0.5*ATR and turns up (oversold bounce)
                # Short: Bull Power crosses above 0.5*ATR and turns down (overbought rejection)
                if (i >= 2 and 
                    bull_power[i-1] < (-0.5 * atr_6h[i-1]) and 
                    bull_power[i] > (-0.5 * atr_6h[i]) and 
                    bull_power[i] > bull_power[i-1] and  # turning up
                    volume_confirm):
                    signals[i] = 0.25
                    position = 1
                elif (i >= 2 and 
                      bull_power[i-1] > (0.5 * atr_6h[i-1]) and 
                      bull_power[i] < (0.5 * atr_6h[i]) and 
                      bull_power[i] < bull_power[i-1] and  # turning down
                      volume_confirm):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if is_trending:
                # Exit long when Bear Power turns down (trend weakening)
                if (i >= 2 and 
                    bear_power[i-2] > 0 and bear_power[i-1] > 0 and bear_power[i] < bear_power[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                # Exit long when Bull Power crosses above zero (mean reversion complete)
                if bull_power[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            if is_trending:
                # Exit short when Bull Power turns up (trend weakening)
                if (i >= 2 and 
                    bull_power[i-2] < 0 and bull_power[i-1] < 0 and bull_power[i] > bull_power[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                # Exit short when Bull Power crosses below zero (mean reversion complete)
                if bull_power[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals