#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d ADX regime filter and volume confirmation.
# Long when Elder Bull Power > 0 AND 1d ADX > 25 (trending market) AND 6h volume > 1.5x 20-period volume MA.
# Short when Elder Bear Power < 0 AND 1d ADX > 25 (trending market) AND 6h volume > 1.5x 20-period volume MA.
# Exit when Elder Power crosses zero OR ADX < 20 (range market) OR volume normalizes.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year).
# Elder Ray measures trend strength via bull/bear power relative to EMA13, ADX filters for trending regimes only,
# volume confirms institutional participation. Works in both bull and bear by only trading strong trends.

name = "6h_ElderRay_ADX_Regime_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength regime
    # ADX requires +DI, -DI, and TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Regime conditions: trending market (ADX > 25)
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20  # Exit condition for ranging
        
        # Volume spike condition: current 6h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma[i] * 1.5)
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0  # Bull power positive
        bear_strong = bear_power[i] < 0  # Bear power negative
        
        if position == 0:
            # Enter long: bull power positive AND trending AND volume spike
            if bull_strong and trending and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: bear power negative AND trending AND volume spike
            elif bear_strong and trending and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power turns negative OR ranging market OR volume normalizes
            if not bull_strong or ranging or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power turns positive OR ranging market OR volume normalizes
            if not bear_strong or ranging or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals