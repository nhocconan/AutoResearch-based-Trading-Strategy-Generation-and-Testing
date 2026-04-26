#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Regime_ADX
Hypothesis: On 12h timeframe, Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike (>1.5x 20-period average), and ADX regime filter (ADX>25) captures sustainable trends while avoiding whipsaws. Long when price breaks above R3 with volume spike and bullish alignment; Short when price breaks below S3 with volume spike and bearish alignment. Uses discrete sizing (±0.25) to minimize fee drag and works in both bull/bear markets with BTC/ETH edge. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (using typical price)
    typical_price = (high + low + close) / 3
    # For 12h timeframe, we need previous day's typical price range
    # We'll use rolling window of 2 periods (24h) to get previous day's HLC
    tp_series = pd.Series(typical_price)
    # Shift by 2 to get previous day's typical price (2*12h = 24h)
    tp_prev = tp_series.shift(2)
    high_prev = pd.Series(high).shift(2)
    low_prev = pd.Series(low).shift(2)
    close_prev = pd.Series(close).shift(2)
    
    # Calculate pivot and ranges from previous day
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    r3 = pivot + (range_prev * 1.1 / 4)
    s3 = pivot - (range_prev * 1.1 / 4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    # ADX calculation (12h)
    # True Range
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high).diff()
    down_move = -(pd.Series(low).diff())  # negative of low diff
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of typical price calculation (2), volume MA (20), ADX (14*2=28), 1d EMA34
    start_idx = max(2, 20, 28) + 2  # +2 for previous day data
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx[i]
        pdi = plus_di[i]
        mdi = minus_di[i]
        ema_34_val = ema_34_1d_aligned[i]
        
        # Regime conditions: ADX > 25 for trending market
        strong_trend = adx_val > 25
        
        # Entry conditions
        long_entry = (close_val > r3[i]) and vol_spike and strong_trend and (pdi > mdi) and (close_val > ema_34_val)
        short_entry = (close_val < s3[i]) and vol_spike and strong_trend and (mdi > pdi) and (close_val < ema_34_val)
        
        # Exit conditions: reverse signal or trend deterioration
        long_exit = (close_val < pivot[i]) or (adx_val < 20) or (mdi > pdi) or (close_val < ema_34_val)
        short_exit = (close_val > pivot[i]) or (adx_val < 20) or (pdi > mdi) or (close_val > ema_34_val)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Regime_ADX"
timeframe = "12h"
leverage = 1.0