#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot + Volume Spike + ADX Trend Filter
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# Long when price crosses above L3 with volume spike and ADX > 20 (trending).
# Short when price crosses below H3 with volume spike and ADX > 20.
# Works in bull via L3 breakouts + volume + uptrend, in bear via H3 breakdowns + volume + downtrend.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_camarilla_pivot_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # H4 = C + (H-L) * 1.1/2
    # H3 = C + (H-L) * 1.1/4
    # L3 = C - (H-L) * 1.1/4
    # L4 = C - (H-L) * 1.1/2
    # where C, H, L are previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use only previous day's data
    H4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    H3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    L3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    L4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 4h timeframe (already shift(1) inside for previous day)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    
    # ADX trend filter (14-period) on 4h
    # Calculate +DM, -DM, TR
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(low, prepend=low[0]) * -1  # positive when low decreases
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr1[0] = 0  # first TR has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(values, period):
        smoothed = np.zeros_like(values)
        if len(values) == 0:
            return smoothed
        smoothed[period-1] = np.nansum(values[:period])  # seed
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    period = 14
    atr_period = wilders_smooth(tr, period)
    plus_di = 100 * wilders_smooth(plus_dm, period) / (atr_period + 1e-10)
    minus_di = 100 * wilders_smooth(minus_dm, period) / (atr_period + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smooth(dx, period)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(period, n):
        # Skip if required data not available
        if (np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price below L3 OR ADX < 20 (trend weak)
            if close[i] < L3_4h[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price above H3 OR ADX < 20 (trend weak)
            if close[i] > H3_4h[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and adx[i] >= 20:
                # Long: price crosses above L3 with volume spike
                if close[i] > L3_4h[i] and (i == period or close[i-1] <= L3_4h[i-1]):
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below H3 with volume spike
                elif close[i] < H3_4h[i] and (i == period or close[i-1] >= H3_4h[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals