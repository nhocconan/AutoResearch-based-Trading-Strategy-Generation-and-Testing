#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND 6h volume > 1.3x 20-period volume MA.
# Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (trending) AND 6h volume > 1.3x 20-period volume MA.
# Exit when Elder Ray divergence occurs OR 1d ADX < 20 (range) OR volume drops.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Elder Ray measures bull/bear power via EMA13, 1d ADX filters for trending regimes only, volume confirms participation.
# Works in both bull and bear markets by only trading strong trends when volume confirms and Elder Ray shows clear dominance.

name = "6h_ElderRay_1dADX_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength
    # ADX: Average Directional Index - measures trend strength (not direction)
    # +DI: Positive Directional Indicator
    # -DI: Negative Directional Indicator
    # DX = |(+DI - -DI)| / (+DI + -DI) * 100
    # ADX = smoothed DX
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # +DM and -DM
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = atr.values  # already smoothed
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray on 6h timeframe
    # Elder Ray: Bull Power = High - EMA13(Close)
    #            Bear Power = Low - EMA13(Close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 6h volume > 1.3x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_6h[i] * 1.3)
        
        # Elder Ray conditions
        bull_power_pos = bull_power[i] > 0    # Bulls in control
        bear_power_neg = bear_power[i] < 0    # Bears weak
        bear_power_pos = bear_power[i] > 0    # Bears in control
        bull_power_neg = bull_power[i] < 0    # Bulls weak
        
        # 1d ADX trend condition
        trending = adx_1d_aligned[i] > 25     # Strong trend
        ranging = adx_1d_aligned[i] < 20      # Range/market weak
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND trending AND volume spike AND session
            if bull_power_pos and bear_power_neg and trending and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND trending AND volume spike AND session
            elif bear_power_pos and bull_power_neg and trending and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Elder Ray divergence OR ranging OR volume drops
            if not (bull_power_pos and bear_power_neg) or ranging or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Elder Ray divergence OR ranging OR volume drops
            if not (bear_power_pos and bull_power_neg) or ranging or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals