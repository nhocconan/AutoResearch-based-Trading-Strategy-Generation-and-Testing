#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h ADX trend filter and 1d volume confirmation
# - Long when: 1h RSI(14) < 30, 4h ADX > 25 (trending), 1d volume > 1.5x 20-day average
# - Short when: 1h RSI(14) > 70, 4h ADX > 25 (trending), 1d volume > 1.5x 20-day average
# - Exit when RSI returns to neutral range (40-60) or ADX < 20 (trend weak)
# - Time-based exit: force close after 12 bars to prevent overstay
# - Session filter: only trade 08-20 UTC to avoid low-liquidity hours
# - Designed to work in both bull and bear by using ADX to confirm trend strength
# - Target: 25-35 trades/year (~100-140 total over 4 years) to minimize fee drag

name = "1h_RSI_ADX_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-calculate hours for session filter
    hours = prices.index.hour
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate ADX(14) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_values(x, period):
        if len(x) < period:
            return np.full_like(x, np.nan)
        smoothed = np.zeros_like(x)
        smoothed[period-1] = np.nansum(x[:period])
        for i in range(period, len(x)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + x[i]
        return smoothed
    
    period = 14
    atr = smooth_values(tr, period)
    dm_plus_smooth = smooth_values(dm_plus, period)
    dm_minus_smooth = smooth_values(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_values(dx, period)
    
    # Align ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1h RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any required data is NaN
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
                bars_since_entry = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current 1h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 1h: 1d has 24x 1h bars, so divide by 24
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 24.0)
        
        # ADX filter: trending market
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for entries only in session with volume and trend
            if in_session and volume_filter and strong_trend:
                # Long when oversold
                if rsi[i] < 30:
                    signals[i] = 0.20
                    position = 1
                    bars_since_entry = 0
                # Short when overbought
                elif rsi[i] > 70:
                    signals[i] = -0.20
                    position = -1
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # Long position: exit conditions
            exit_condition = (
                rsi[i] > 40 or  # RSI recovered from oversold
                weak_trend or   # Trend weakening
                bars_since_entry >= 12  # Time-based exit
            )
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit conditions
            exit_condition = (
                rsi[i] < 60 or  # RSI recovered from overbought
                weak_trend or   # Trend weakening
                bars_since_entry >= 12  # Time-based exit
            )
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
    
    return signals