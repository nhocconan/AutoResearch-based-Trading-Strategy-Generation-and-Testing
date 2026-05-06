#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
# Long when price breaks above 1w Camarilla R3 level AND 1d volume > 2.0 * avg_volume(20) AND 1d ADX > 25
# Short when price breaks below 1w Camarilla S3 level AND 1d volume > 2.0 * avg_volume(20) AND 1d ADX > 25
# Exit when price crosses 1w Camarilla pivot level (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1w Camarilla provides weekly structure with proven R3/S3 breakout edge
# Volume spike confirms institutional participation (reduces false breakouts)
# ADX > 25 ensures we only trade in trending markets (works in bull/bear regimes)

name = "6h_1wCamarillaR3S3_1dVolumeSpike_ADX_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need sufficient data for pivots
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels based on previous 1w bar
    # Camarilla: R3 = Close + 1.125 * (High - Low), S3 = Close - 1.125 * (High - Low)
    camarilla_r3_1w = close_1w + 1.125 * (high_1w - low_1w)
    camarilla_s3_1w = close_1w - 1.125 * (high_1w - low_1w)
    camarilla_pivot_1w = (high_1w + low_1w + close_1w) / 3.0  # Standard pivot for exit
    
    # Get 1d data ONCE before loop for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX and volume average
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    
    # Calculate 1d ADX for regime filter (trending market detection)
    # ADX calculation: +DI, -DI, DX, then ADX smoothed
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # First period has no prior close
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed averages
        def smooth(values, period):
            result = np.full_like(values, np.nan, dtype=float)
            if len(values) < period:
                return result
            # First value: simple average
            result[period-1] = np.nansum(values[:period]) / period
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
            return result
        
        atr = smooth(tr, period)
        plus_di = 100 * smooth(plus_dm, period) / atr
        minus_di = 100 * smooth(minus_dm, period) / atr
        
        # DX and ADX
        dx = np.full_like(close, np.nan, dtype=float)
        mask = (plus_di + minus_di) > 0
        dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
        
        adx = smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    adx_filter_1d = adx_1d > 25  # Trending market regime
    
    # Align 1w Camarilla levels to 6h timeframe (wait for completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot_1w)
    
    # Align 1d indicators to 6h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(adx_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R3 with volume spike and ADX > 25
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                volume_spike_aligned[i] and adx_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S3 with volume spike and ADX > 25
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  volume_spike_aligned[i] and adx_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals