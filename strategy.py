#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume-Weighted MACD with 1d ADX trend filter and RSI momentum filter.
# Uses MACD histogram weighted by volume for stronger signals, combined with 1d ADX > 25
# for trend strength and RSI(14) for momentum confirmation. Designed to capture strong
# momentum moves in both bull and bear markets while avoiding false signals in low-volume
# or sideways conditions. Target: 50-150 total trades over 4 years (12-37/year).
name = "4h_VolMACD_1dADX25_RSI14"
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 14-period ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smoothed_avg(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(x[1:period])
        # Subsequent values are smoothed
        for i in range(period, len(x)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr = smoothed_avg(tr, 14)
    dm_plus_smooth = smoothed_avg(dm_plus, 14)
    dm_minus_smooth = smoothed_avg(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_avg(dx, 14)
    adx_14_1d = adx  # Already calculated on 1d
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Volume-weighted MACD on 4h data
    # Calculate EMA12 and EMA26 of close
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_12 - ema_26
    
    # Signal line: EMA9 of MACD line
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # MACD histogram
    macd_hist = macd_line - signal_line
    
    # Volume-weighted MACD histogram
    # Normalize volume to [0,1] range for weighting
    vol_max = np.maximum.reduce([np.nanmax(volume[i-49:i+1]) if i >= 49 else np.nanmax(volume[:i+1]) for i in range(len(volume))])
    vol_min = np.minimum.reduce([np.nanmin(volume[i-49:i+1]) if i >= 49 else np.nanmin(volume[:i+1]) for i in range(len(volume))])
    vol_range = vol_max - vol_min
    vol_norm = np.where(vol_range != 0, (volume - vol_min) / vol_range, 0.5)
    vol_weighted_hist = macd_hist * (1 + vol_norm)  # Amplify during high volume
    
    # RSI(14) for momentum
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        if len(close_prices) < period:
            return avg_gain
            
        # First average
        avg_gain[period] = np.nanmean(gain[1:period+1])
        avg_loss[period] = np.nanmean(loss[1:period+1])
        
        # Subsequent averages
        for i in range(period+1, len(close_prices)):
            if not np.isnan(avg_gain[i-1]):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 14)  # Need enough data for MACD and RSI
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_weighted_hist[i]) or 
            np.isnan(rsi_14[i]) or np.isnan(signal_line[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_14_1d_aligned[i]
        vwmacd = vol_weighted_hist[i]
        signal_val = signal_line[i]
        rsi_val = rsi_14[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        vol_spike = vol > 1.5 * vol_ma  # Volume at least 1.5x average
        
        if position == 0:
            # Enter long: VMACD histogram > 0 AND signal line > 0 (bullish momentum) 
            # AND ADX > 25 (strong trend) AND RSI > 50 (bullish momentum) AND volume spike
            if vwmacd > 0 and signal_val > 0 and adx_val > 25 and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: VMACD histogram < 0 AND signal line < 0 (bearish momentum)
            # AND ADX > 25 (strong trend) AND RSI < 50 (bearish momentum) AND volume spike
            elif vwmacd < 0 and signal_val < 0 and adx_val > 25 and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VMACD histogram < 0 OR signal line < 0 (momentum fading)
            # OR ADX < 20 (trend weakening) OR RSI < 40 (losing momentum)
            if vwmacd < 0 or signal_val < 0 or adx_val < 20 or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VMACD histogram > 0 OR signal line > 0 (momentum fading)
            # OR ADX < 20 (trend weakening) OR RSI > 60 (losing momentum)
            if vwmacd > 0 or signal_val > 0 or adx_val < 20 or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals