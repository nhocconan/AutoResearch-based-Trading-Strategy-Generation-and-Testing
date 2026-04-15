#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) with ADX regime filter
# Long when: Alligator bullish alignment (jaw < teeth < lips) + Bull Power > 0 + ADX > 25 (trending)
# Short when: Alligator bearish alignment (jaw > teeth > lips) + Bear Power < 0 + ADX > 25 (trending)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Alligator identifies trend structure, Elder Ray measures bull/bear power behind the move,
# ADX ensures we only trade in trending markets to avoid whipsaws in ranging conditions.
# Target: 12-37 trades/year on 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Elder Ray (Bull Power/Bear Power) and ADX ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema_13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema_13_1d
    
    # ADX calculation (14-period)
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth / atr_1d
    minus_di_1d = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Williams Alligator ===
    # Alligator lines: Jaw (13-period SMMA, 8-period offset), Teeth (8-period SMMA, 5-period offset), Lips (5-period SMMA, 3-period offset)
    # Using EMA as approximation for SMMA (common practice)
    ema_5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Alligator with offsets (shifted forward)
    lips = np.roll(ema_5, 3)   # 5-period EMA, 3-period offset
    teeth = np.roll(ema_8, 5)  # 8-period EMA, 5-period offset
    jaw = np.roll(ema_13, 8)   # 13-period EMA, 8-period offset
    
    # Handle NaN from roll
    lips[:3] = np.nan
    teeth[:5] = np.nan
    jaw[:8] = np.nan
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 13, 8, 5) + 10  # EMA periods + ADX + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: jaw < teeth < lips
        # 2. Bull Power > 0 (bulls in control)
        # 3. ADX > 25 (strong trend)
        if (jaw[i] < teeth[i]) and (teeth[i] < lips[i]) and \
           (bull_power_1d_aligned[i] > 0) and (adx_1d_aligned[i] > 25):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: jaw > teeth > lips
        # 2. Bear Power < 0 (bears in control)
        # 3. ADX > 25 (strong trend)
        elif (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and \
             (bear_power_1d_aligned[i] < 0) and (adx_1d_aligned[i] > 25):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Alligator_ElderRay_ADX_Regime_Filter_v1"
timeframe = "6h"
leverage = 1.0