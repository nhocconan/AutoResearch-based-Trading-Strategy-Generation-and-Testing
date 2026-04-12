#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w volume confirmation and ADX trend filter
# Works in bull/bear: breakouts capture trends, volume confirms institutional interest,
# ADX filter avoids whipsaws in ranging markets. Target: 10-25 trades/year.
name = "1d_1w_donchian_breakout_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w volume spike: volume > 2.0x 10-week average
    vol_ma_1w = pd.Series(df_1w['volume']).rolling(window=10, min_periods=10).mean().values
    vol_spike_1w = df_1w['volume'] > (vol_ma_1w * 2.0)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # 1d Donchian channels (20-day)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX trend filter on 1d
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros(n)
    dm_plus_smooth = np.zeros(n)
    dm_minus_smooth = np.zeros(n)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # DI and DX
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    
    # ADX: smoothed DX
    adx = np.zeros(n)
    adx[2*period-1] = np.mean(dx[period-1:2*period-1])
    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # ADX filter: trend present when ADX > 25
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1w_aligned[i]) or np.isnan(adx_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above Donchian high with volume spike and ADX filter
        long_signal = close[i] > donchian_high[i] and vol_spike_1w_aligned[i] and adx_filter[i]
        # Short: break below Donchian low with volume spike and ADX filter
        short_signal = close[i] < donchian_low[i] and vol_spike_1w_aligned[i] and adx_filter[i]
        
        # Exit when price returns to midpoint of Donchian channel
        donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
        exit_long = close[i] < donchian_mid
        exit_short = close[i] > donchian_mid
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals