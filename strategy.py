# 4h Camarilla Pivot Breakout with Volume Spike and ADX Trend Filter
# Strategy: Trade Camarilla pivot level breakouts (R1/S1) with volume confirmation.
# Use ADX to filter trades in trending markets (ADX > 25) to avoid choppy whipsaws.
# Targets 30-50 trades per year to minimize fee drag while capturing strong momentum.
# Works in both bull and bear markets by following established trends.
# Timeframe: 4h (primary), HTF: 1d for pivots and trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_r1 = prev_close + range_ * 1.1 / 12
    camarilla_s1 = prev_close - range_ * 1.1 / 12
    camarilla_r2 = prev_close + range_ * 1.1 / 6
    camarilla_s2 = prev_close - range_ * 1.1 / 6
    
    # Calculate ADX for trend filtering
    # +DI and -DI calculation
    high_diff = df_1d['high'].diff()
    low_diff = df_1d['low'].diff().multiply(-1)
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smooth(tr.values, period)
    plus_di_1d = 100 * wilders_smooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilders_smooth(minus_dm, period) / atr_1d
    dx_1d = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smooth(dx_1d, period)
    
    # Align daily data to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_r2_aligned[i]) or
            np.isnan(camarilla_s2_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r2 = camarilla_r2_aligned[i]
        s2 = camarilla_s2_aligned[i]
        adx = adx_1d_aligned[i]
        
        # Only trade in trending markets (ADX > 25)
        if adx <= 25:
            # In chop, stay flat or reduce position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike
            if price > r1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike
            elif price < s1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below S1 or reverses below R1
            if price < s1 or price < r1:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above R1 or reverses above S1
            if price > r1 or price > s1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_PivotBreakout_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0