#!/usr/bin/env python3
"""
Hypothesis: 6h ATR breakout with 12h ADX trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for ADX(14) trend filter (strong trend >25).
- Entry: Long when price breaks above ATR(14) upper band AND 12h ADX>25 AND volume > 1.5 * 6h volume MA(20);
         Short when price breaks below ATR(14) lower band AND 12h ADX>25 AND volume > 1.5 * 6h volume MA(20).
- ATR bands: upper = SMA(20, close) + 2.0 * ATR(14), lower = SMA(20, close) - 2.0 * ATR(14).
- Exit: Close-based reversal (opposite signal) or stoploss via ATR trailing (signal=0 when price closes below/above 20-period SMA).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR breakouts capture volatility expansion; 12h ADX ensures we only trade strong trends, avoiding whipsaws in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # first value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/14)
    def wilder_smoothing(x, period):
        x = x.astype(float)
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            # first value is simple average
            result[period-1] = np.nansum(x[:period])
            # subsequent values: Wilder's smoothing
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr_12h = wilder_smoothing(tr, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smoothing(dx, 14)
    
    # Get 6h data for ATR bands and volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate ATR(14) on 6h for bands
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = 0
    
    atr_6h = wilder_smoothing(tr_6h, 14)
    
    # Calculate SMA(20) on 6h close for bands midpoint
    sma_20_6h = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR bands: upper = SMA(20) + 2.0 * ATR(14), lower = SMA(20) - 2.0 * ATR(14)
    upper_band = sma_20_6h + 2.0 * atr_6h
    lower_band = sma_20_6h - 2.0 * atr_6h
    
    # Calculate volume MA(20) on 6h
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    upper_band_aligned = align_htf_to_ltf(prices, df_6h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_6h, lower_band)
    sma_20_aligned = align_htf_to_ltf(prices, df_6h, sma_20_6h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (max of 28 for ADX, 20 for bands/vol MA)
    start_idx = max(28, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(sma_20_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above 20-period SMA
        if position == 1:
            if curr_close < sma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > sma_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation and trend filter
        bullish_breakout = curr_close > upper_band_aligned[i]
        bearish_breakout = curr_close < lower_band_aligned[i]
        
        # Trend filter from 12h ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if strong_trend and vol_confirm:
                # Long: bullish breakout AND strong trend
                if bullish_breakout:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish breakout AND strong trend
                elif bearish_breakout:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ATR_Breakout_12hADX25_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0