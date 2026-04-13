#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and ADX regime filter
    # Long: price breaks above H3 pivot AND volume > 1.5x 20-period avg AND ADX > 25 (trending)
    # Short: price breaks below L3 pivot AND volume > 1.5x 20-period avg AND ADX > 25 (trending)
    # Exit: price re-enters H3-L3 range OR volume dry-up OR ADX < 20 (range)
    # Using 4h timeframe for optimal trade frequency, Camarilla for intraday structure,
    # 1d volume for confirmation, ADX for trend regime (avoid whipsaws in ranging markets).
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3 = pp + (range_1d * 1.1 / 4)
    l3 = pp - (range_1d * 1.1 / 4)
    h4 = pp + (range_1d * 1.1 / 2)
    l4 = pp - (range_1d * 1.1 / 2)
    
    # Align daily Camarilla levels to 4h (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate daily volume for confirmation (>1.5x 20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    volume_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 4h ADX(14) for regime filter
    # ADX requires +DI, -DI, and TR
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, high - prev_close, prev_close - low)
    # +DI = 100 * smoothed(+DM) / smoothed(TR)
    # -DI = 100 * smoothed(-DM) / smoothed(TR)
    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    # ADX = smoothed(DX)
    
    # Calculate +DM and -DM
    high_diff = high[1:] - high[:-1]
    low_diff = low[:-1] - low[1:]
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Pad with zeros for first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Calculate True Range (TR)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    # Smoothed +DM, -DM, and TR
    smoothed_plus_dm = wilders_smoothing(plus_dm, period)
    smoothed_minus_dm = wilders_smoothing(minus_dm, period)
    smoothed_tr = wilders_smoothing(tr, period)
    
    # Calculate +DI and -DI
    plus_di = np.where(smoothed_tr != 0, 100 * smoothed_plus_dm / smoothed_tr, 0)
    minus_di = np.where(smoothed_tr != 0, 100 * smoothed_minus_dm / smoothed_tr, 0)
    
    # Calculate DX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Calculate ADX (smoothed DX)
    adx = wilders_smoothing(dx, period)
    
    # Get 4h volume for additional confirmation
    vol_ma_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_4h[i] = np.mean(volume[i-20:i])
    volume_spike_4h = volume > (1.5 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending market
        trending = adx[i] > 25
        ranging = adx[i] < 20  # Exit condition for ranging
        
        # Volume confirmation (either 1d or 4h spike)
        vol_confirm = volume_spike_1d_aligned[i] or volume_spike_4h[i]
        
        # Entry logic: Camarilla breakout + volume confirmation + trending regime
        long_entry = (close[i] > h3_aligned[i]) and vol_confirm and trending
        short_entry = (close[i] < l3_aligned[i]) and vol_confirm and trending
        
        # Exit logic: re-enter H3-L3 range OR volume dry-up OR ranging market
        long_exit = (close[i] < l3_aligned[i]) or not vol_confirm or ranging
        short_exit = (close[i] > h3_aligned[i]) or not vol_confirm or ranging
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0