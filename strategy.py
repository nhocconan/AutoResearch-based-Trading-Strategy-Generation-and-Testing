#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX regime filter with volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND ADX(14) > 25 AND volume > 1.5x 20-period average
# - Short when Bear Power > 0 AND Bull Power < 0 AND ADX(14) > 25 AND volume > 1.5x 20-period average
# - Exit when Bull Power and Bear Power have same sign (both positive or both negative) OR ADX < 20
# - Uses 1d HTF for trend filter: only take longs when price > 1d EMA(50), shorts when price < 1d EMA(50)
# - Discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear power relative to EMA
# - ADX filters for trending markets (avoids chop)
# - 1d EMA ensures we trade with higher timeframe trend
# - Volume confirmation reduces false signals

name = "6h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h Elder Ray components
    bull_power = high - ema13  # High - EMA(13)
    bear_power = ema13 - low   # EMA(13) - Low
    
    # Pre-compute 6h ADX(14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    tr_smooth = np.zeros_like(tr)
    
    # First values (simple sum)
    plus_dm_smooth[13] = np.sum(plus_dm[1:14])
    minus_dm_smooth[13] = np.sum(minus_dm[1:14])
    tr_smooth[13] = np.sum(tr[1:14])
    
    # Subsequent values (Wilder's smoothing)
    for i in range(14, len(tr)):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + tr[i]
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.zeros_like(plus_di)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX value
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i]) or np.isnan(ema50_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND price > 1d EMA(50) AND volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx[i] > 25 and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 AND price < 1d EMA(50) AND volume spike
            elif (bear_power[i] > 0 and 
                  bull_power[i] < 0 and 
                  adx[i] > 25 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Bull Power and Bear Power same sign (both positive or both negative) OR ADX < 20
            exit_long = (position == 1 and 
                        ((bull_power[i] > 0 and bear_power[i] > 0) or  # Both positive
                         (bull_power[i] < 0 and bear_power[i] < 0) or  # Both negative
                         adx[i] < 20))
            exit_short = (position == -1 and 
                         ((bull_power[i] > 0 and bear_power[i] > 0) or  # Both positive
                          (bull_power[i] < 0 and bear_power[i] < 0) or  # Both negative
                          adx[i] < 20))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals