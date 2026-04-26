#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime_ADX
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike (>2.0x 20-bar MA), and ADX regime filter (ADX>25 for trending markets). This combines proven Camarilla structure with trend alignment and volatility filtering to reduce false breakouts. Designed for 20-40 trades/year (80-160 total over 4 years) to minimize fee drag while maintaining edge in both bull and bear markets via trend-following structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla levels (using 1d data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3
    # R3 = Close + ((High-Low) * 1.1/4)
    # S3 = Close - ((High-Low) * 1.1/4)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + (rng * 1.1 / 4)
    camarilla_s3 = close_1d - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (they change only at 1d boundaries)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ADX regime filter: only trade when ADX > 25 (trending market)
    # Calculate ADX using 14-period Wilder's smoothing
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original arrays
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[:period])
                # Subsequent values: Wilder's smoothing
                for i in range(period, len(data)):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilder_smooth(tr, period)
        smoothed_plus_dm = wilder_smooth(plus_dm, period)
        smoothed_minus_dm = wilder_smooth(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * smoothed_plus_dm / atr
        minus_di = 100 * smoothed_minus_dm / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period)
        
        return adx
    
    adx_values = calculate_adx(high, low, close, period=14)
    adx_filter = adx_values > 25  # Only trade in trending markets (ADX > 25)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 34 for ema, 14*3 for adx, 1 for camarilla)
    start_idx = max(20, 34, 14*3)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(adx_values[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        adx_ok = adx_filter[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions: breakout of Camarilla R3/S3 in trend direction with volume spike and ADX filter
        long_entry = (close_val > camarilla_r3_val) and bullish_1d and vol_spike and adx_ok
        short_entry = (close_val < camarilla_s3_val) and bearish_1d and vol_spike and adx_ok
        
        # Exit conditions: opposite Camarilla level touch (S3 for long, R3 for short)
        exit_long = close_val < camarilla_s3_val
        exit_short = close_val > camarilla_r3_val
        
        # Minimum holding period: 4 bars (to avoid whipsaw)
        min_hold = 4
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime_ADX"
timeframe = "4h"
leverage = 1.0