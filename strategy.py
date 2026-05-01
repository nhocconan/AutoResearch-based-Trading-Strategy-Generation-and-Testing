#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend filter (EMA20) and 1d regime filter (ADX) for signal direction,
# with 1h Donchian(20) breakout for entry timing and volume confirmation.
# Long when: price breaks above 1h Donchian upper band AND 4h EMA20 rising AND 1d ADX > 25 (trending) AND volume > 1.5x 20-bar average.
# Short when: price breaks below 1h Donchian lower band AND 4h EMA20 falling AND 1d ADX > 25 (trending) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.20 to minimize fee churn. Session filter (08-20 UTC) reduces noise trades.
# Designed for 1h timeframe to capture medium-term trends with tight entries.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_Donchian20_4hEMA20_1dADX_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 4h data ONCE before loop for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA20 calculation
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 4h EMA20 slope (rising/falling)
    ema_20_4h_slope = np.diff(ema_20_4h_aligned, prepend=ema_20_4h_aligned[0])
    ema_20_4h_rising = ema_20_4h_slope > 0
    ema_20_4h_falling = ema_20_4h_slope < 0
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = -pd.Series(low_1d).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed DM
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current 1h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA, ADX and Donchian calculation
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_20_4h_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        trending_regime = adx_aligned[i] > 25  # Only trade in trending markets
        
        # Donchian breakout signals
        breakout_up = curr_high > highest_high[i]  # break above upper band
        breakout_down = curr_low < lowest_low[i]   # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper band AND 4h EMA20 rising AND trending regime AND volume confirmation
            if (breakout_up and 
                ema_20_4h_rising[i] and 
                trending_regime and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: breakout below lower band AND 4h EMA20 falling AND trending regime AND volume confirmation
            elif (breakout_down and 
                  ema_20_4h_falling[i] and 
                  trending_regime and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (stoploss) OR 4h EMA20 falls (trend change) OR ADX < 20 (regime change to range)
            if (curr_low < lowest_low[i] or 
                ema_20_4h_falling[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (stoploss) OR 4h EMA20 rises (trend change) OR ADX < 20 (regime change to range)
            if (curr_high > highest_high[i] or 
                ema_20_4h_rising[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals