#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
# Long when: Bear Power < 0 (bullish) AND Bull Power rising AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period average
# Short when: Bull Power > 0 (bearish) AND Bear Power falling AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 12-37 trades/year on 6h.
# Elder Ray measures bull/bear power via EMA13, ADX filters for trending regimes only, volume spike confirms conviction.
# Works in bull (catching strong uptrends) and bear (catching strong downtrends) by trading with the trend.

name = "6h_ElderRay_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for Elder Ray (EMA13 of high/low)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Elder Ray on 6h: EMA13 of high and low
    ema13_high = pd.Series(df_6h['high'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_low = pd.Series(df_6h['low'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = df_6h['high'].values - ema13_high  # Bull Power: High - EMA13(High)
    bear_power = df_6h['low'].values - ema13_low    # Bear Power: Low - EMA13(Low)
    
    # Align Elder Ray components to 6h primary timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # 1d ADX calculation (14-period)
    # True Range
    tr1 = pd.Series(df_1d['high'].values - df_1d['low'].values)
    tr2 = pd.Series(np.abs(df_1d['high'].values - df_1d['close'].shift(1).values))
    tr3 = pd.Series(np.abs(df_1d['low'].values - df_1d['close'].shift(1).values))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = df_1d['high'].values - df_1d['high'].shift(1).values
    down_move = df_1d['low'].shift(1).values - df_1d['low'].values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h primary timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h volume average (20-period) for volume spike confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA and Elder Ray
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        curr_adx = adx_1d_aligned[i]
        
        # Volume spike: current 6h volume > 1.5x 20-period average
        volume_spike = curr_vol > (curr_vol_ma * 1.5)
        
        # Elder Ray conditions
        bullish_cond = (curr_bear_power < 0) and (curr_bull_power > bull_power_aligned[max(0, i-1)])  # Bear Power < 0 AND Bull Power rising
        bearish_cond = (curr_bull_power > 0) and (curr_bear_power < bear_power_aligned[max(0, i-1)])  # Bull Power > 0 AND Bear Power falling
        
        # 1d trend filter: ADX > 25 indicates trending market
        trending = curr_adx > 25.0
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Bear Power < 0 AND Bull Power rising AND trending AND volume spike
            if (bullish_cond and 
                trending and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power > 0 AND Bear Power falling AND trending AND volume spike
            elif (bearish_cond and 
                  trending and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear Power becomes positive OR Bull Power stops rising
            if (curr_bear_power >= 0 or 
                curr_bull_power <= bull_power_aligned[max(0, i-1)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power becomes negative OR Bear Power stops falling
            if (curr_bull_power <= 0 or 
                curr_bear_power >= bear_power_aligned[max(0, i-1)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals