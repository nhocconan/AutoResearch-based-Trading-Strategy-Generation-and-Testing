#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + ADX trend filter + volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND ADX > 25 (trending up) AND volume > 1.2x 20-period average
# - Short when Bear Power > 0 AND ADX > 25 (trending down) AND volume > 1.2x 20-period average
# - Exit when power reverses sign OR ADX < 20 (trend weakens)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures bull/bear strength behind price moves
# - ADX filters for trending markets where Elder Ray works best
# - Volume confirmation ensures trend has participation

name = "6h_1d_elder_ray_adx_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power and Bear Power
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # ADX(14) calculation
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - np.roll(close, 1)[1:])
    tr3 = np.abs(low[1:] - np.roll(close, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first element is NaN
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.2 * vol_ma)
    
    # Pre-compute 1d trend filter (EMA50 > EMA200 for uptrend, < for downtrend)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_uptrend = ema_50_1d > ema_200_1d
    daily_downtrend = ema_50_1d < ema_200_1d
    
    # Align HTF indicators to 6h timeframe
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend)
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND ADX > 25 (strong uptrend) AND volume spike AND daily uptrend
            if (bull_power[i] > 0 and 
                adx[i] > 25 and 
                volume_spike[i] and 
                daily_uptrend_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power > 0 AND ADX > 25 (strong downtrend) AND volume spike AND daily downtrend
            elif (bear_power[i] > 0 and 
                  adx[i] > 25 and 
                  volume_spike[i] and 
                  daily_downtrend_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: power reverses OR ADX < 20 (trend weakening) OR loss of daily trend alignment
            exit_long = (position == 1 and 
                        (bull_power[i] <= 0 or 
                         adx[i] < 20 or 
                         not daily_uptrend_aligned[i]))
            exit_short = (position == -1 and 
                         (bear_power[i] <= 0 or 
                          adx[i] < 20 or 
                          not daily_downtrend_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals