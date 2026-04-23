#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H4/L4 breakout with 1d ADX trend filter and volume confirmation.
- Camarilla H4/L4 levels act as stronger breakout levels than H3/L3 (less noise)
- Breakout above H4 with volume > 2.0x average signals strong bullish momentum
- Breakdown below L4 with volume > 2.0x average signals strong bearish momentum
- 1d ADX > 25 ensures trades only occur in trending markets (avoid chop/range)
- Discrete position size 0.25 to manage drawdown
- Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years)
- Novel combination: Camarilla breakout + ADX trend filter on higher timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate typical price for Camarilla pivots (using prior bar's OHLC)
    typical_price = (high + low + close) / 3.0
    
    # Shift by 1 to use prior bar's data for pivot calculation (no look-ahead)
    typical_price_shifted = np.roll(typical_price, 1)
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    close_shifted = np.roll(close, 1)
    
    # Set first bar to NaN since we don't have prior bar data
    typical_price_shifted[0] = np.nan
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    close_shifted[0] = np.nan
    
    # Camarilla pivot levels (based on prior bar)
    pivot = (high_shifted + low_shifted + close_shifted) / 3.0
    range_hl = high_shifted - low_shifted
    
    # Resistance/support levels (H4/L4 = 1.1*(H-L)/2 from pivot)
    H4 = pivot + (range_hl * 1.1 / 2.0)  # H4 = pivot + 1.1*(H-L)/2
    L4 = pivot - (range_hl * 1.1 / 2.0)  # L4 = pivot - 1.1*(H-L)/2
    
    # Volume confirmation: > 2.0x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initialize first values
    atr[period-1] = np.nanmean(tr[:period])
    plus_dm_smooth[period-1] = np.nanmean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.nanmean(minus_dm[:period])
    
    # Wilder's smoothing
    for i in range(period, len(tr)):
        atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    # DI values
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(plus_di)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX: EMA of DX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.nanmean(dx[period:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 2*period)  # volume MA, ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H4[i]) or np.isnan(L4[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trend_filter = adx_aligned[i] > 25.0
        
        if position == 0:
            # Long: Close > H4 AND volume confirmation AND trending market
            if close[i] > H4[i] and volume_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: Close < L4 AND volume confirmation AND trending market
            elif close[i] < L4[i] and volume_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < pivot OR ADX drops below 20 (trend weakening)
            if close[i] < pivot[i] or adx_aligned[i] < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > pivot OR ADX drops below 20 (trend weakening)
            if close[i] > pivot[i] or adx_aligned[i] < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_1dADX_TrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0