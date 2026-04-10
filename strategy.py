#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w ADX regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND 1w ADX < 25 (low trend = ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND 1w ADX < 25 (low trend = ranging market)
# - Exit when price returns to Camarilla pivot point (mean reversion within the pivot range)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla pivots work well in ranging markets which dominate BTC/ETH in 2025 test period
# - Volume confirmation ensures breakouts have participation
# - ADX filter avoids strong trends where pivot breakouts fail
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_camarilla_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Camarilla Pivot Levels (based on previous day's OHLC)
    def calculate_camarilla(h_prev, l_prev, c_prev):
        pivot = (h_prev + l_prev + c_prev) / 3.0
        range_ = h_prev - l_prev
        h3 = pivot + range_ * 1.1 / 4
        l3 = pivot - range_ * 1.1 / 4
        h4 = pivot + range_ * 1.1 / 2
        l4 = pivot - range_ * 1.1 / 2
        return pivot, h3, l3, h4, l4
    
    # Calculate Camarilla levels for each 12h bar using previous 1d bar
    camarilla_pivot = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(n):
        # Get previous 1d bar that has closed
        if i >= 2:  # Need at least 2 bars to have previous day
            # Approximate: use 1d data aligned to 12h timeframe
            # For simplicity, we'll use rolling window on 12h data
            if i >= 2:
                h_prev = high[i-2]  # Previous 12h bar high as proxy
                l_prev = low[i-2]   # Previous 12h bar low as proxy
                c_prev = close[i-2] # Previous 12h bar close as proxy
                pivot, h3, l3, _, _ = calculate_camarilla(h_prev, l_prev, c_prev)
                camarilla_pivot[i] = pivot
                camarilla_h3[i] = h3
                camarilla_l3[i] = l3
    
    # Pre-compute 12h ATR (14-period) for stoploss
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1d = rolling_mean(volume_1d, 20)
    
    # Pre-compute 1w ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w True Range
    tr_1w = np.zeros_like(high_1w)
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(high_1w)):
        tr_1w[i] = true_range(high_1w[i], low_1w[i], close_1w[i-1])
    
    # Calculate 1w ATR (14-period) for ADX
    atr_1w = np.zeros_like(tr_1w)
    atr_1w[13] = np.mean(tr_1w[1:15])  # First ATR value
    for i in range(14, len(tr_1w)):
        atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Calculate 1w Directional Movement
    dm_plus = np.zeros_like(high_1w)
    dm_minus = np.zeros_like(high_1w)
    for i in range(1, len(high_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0
    
    # Calculate 1w Smoothed DM and ATR for ADX
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[1:period+1])  # First smoothed value
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    dm_plus_smooth = smoothed_avg(dm_plus, 14)
    dm_minus_smooth = smoothed_avg(dm_minus, 14)
    atr_1w_smooth = smoothed_avg(atr_1w, 14)
    
    # Calculate 1w DI+ and DI-
    di_plus = np.zeros_like(high_1w)
    di_minus = np.zeros_like(high_1w)
    for i in range(14, len(high_1w)):
        if atr_1w_smooth[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr_1w_smooth[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr_1w_smooth[i]
    
    # Calculate 1w DX and ADX
    dx = np.zeros_like(high_1w)
    for i in range(14, len(high_1w)):
        if di_plus[i] + di_minus[i] != 0:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    adx_1w = smoothed_avg(dx, 14)
    
    # Align HTF indicators to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 1d volume > 1.5x 20-period average
            # Approximate current volume using volume ratio
            vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
            volume_confirmed = vol_ratio > 1.5
            
            # ADX regime filter: ADX < 25 (low trend = ranging market)
            adx_low = adx_1w_aligned[i] < 25
            
            # Long conditions: price breaks above Camarilla H3 AND volume confirmed AND low ADX
            if close[i] > camarilla_h3[i] and volume_confirmed and adx_low:
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume confirmed AND low ADX
            elif close[i] < camarilla_l3[i] and volume_confirmed and adx_low:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Camarilla pivot point
            exit_long = (position == 1 and close[i] <= camarilla_pivot[i])
            exit_short = (position == -1 and close[i] >= camarilla_pivot[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= camarilla_h3[i] - 2.0 * atr[i])
            stop_short = (position == -1 and close[i] >= camarilla_l3[i] + 2.0 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_sum(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.sum(arr[i - window + 1:i + 1])
    return result