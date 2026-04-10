#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and ADX trend filter
# - Long when price breaks above Donchian(20) upper band AND 1d volume > 1.3x 20-period average AND ADX > 25 (trending market)
# - Short when price breaks below Donchian(20) lower band AND 1d volume > 1.3x 20-period average AND ADX > 25
# - Exit when price returns to Donchian midpoint (mean reversion to channel center)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts capture strong moves in both bull and bear markets
# - Volume confirmation reduces false breakouts
# - ADX filter ensures we only trade in trending conditions
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)

name = "12h_1d_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Donchian channel (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_band = rolling_max(high, 20)
    lower_band = rolling_min(low, 20)
    mid_band = (upper_band + lower_band) / 2
    
    # Pre-compute 12h ADX (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate True Range
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]  # First bar
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    # Calculate ATR (14-period) using Wilder's smoothing
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.nan
        elif i == 14:
            atr[i] = np.mean(tr[1:15])  # First ATR value
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Calculate +DM and -DM
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Calculate smoothed +DM and -DM (14-period)
    smoothed_plus_dm = np.zeros_like(plus_dm)
    smoothed_minus_dm = np.zeros_like(minus_dm)
    for i in range(len(plus_dm)):
        if i < 14:
            smoothed_plus_dm[i] = np.nan
            smoothed_minus_dm[i] = np.nan
        elif i == 14:
            smoothed_plus_dm[i] = np.sum(plus_dm[1:15])
            smoothed_minus_dm[i] = np.sum(minus_dm[1:15])
        else:
            smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * 13 + plus_dm[i]) / 14
            smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate +DI and -DI
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    for i in range(len(high)):
        if np.isnan(atr[i]) or atr[i] == 0:
            plus_di[i] = np.nan
            minus_di[i] = np.nan
        else:
            plus_di[i] = (smoothed_plus_dm[i] / atr[i]) * 100
            minus_di[i] = (smoothed_minus_dm[i] / atr[i]) * 100
    
    # Calculate DX and ADX (14-period)
    dx = np.zeros_like(high)
    for i in range(len(high)):
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or (plus_di[i] + minus_di[i]) == 0:
            dx[i] = np.nan
        else:
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    adx = np.zeros_like(dx)
    for i in range(len(dx)):
        if i < 27:  # Need 14 for DX + 14 for ADX smoothing
            adx[i] = np.nan
        elif i == 27:
            adx[i] = np.mean(dx[14:28])  # First ADX value
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14  # Wilder's smoothing
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Pre-compute 1d volume confirmation (20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.3 * vol_ma_1d)
    
    # Align HTF indicators to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    mid_band_aligned = align_htf_to_ltf(prices, df_1d, mid_band)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(mid_band_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper band AND volume spike AND ADX > 25 (trending)
            if (close[i] > upper_band_aligned[i] and 
                volume_spike_1d_aligned[i] and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below lower band AND volume spike AND ADX > 25 (trending)
            elif (close[i] < lower_band_aligned[i] and 
                  volume_spike_1d_aligned[i] and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to midpoint (mean reversion)
            exit_long = (position == 1 and close[i] < mid_band_aligned[i])
            exit_short = (position == -1 and close[i] > mid_band_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals