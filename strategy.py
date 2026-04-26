#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Trend_Regime_ADX
Hypothesis: On 12h timeframe, Williams Alligator (jaw=EMA13, teeth=EMA8, lips=EMA5) identifies trending markets when aligned (lips > teeth > jaw for uptrend, lips < teeth < jaw for downtrend). Combined with 1w EMA34 trend filter and ADX regime filter to avoid whipsaws. Long when Alligator bullish aligned + price > 1w EMA34 + ADX > 25; Short when bearish aligned + price < 1w EMA34 + ADX > 25. Uses discrete sizing (±0.25) and close-based stops. Designed for 12-37 trades/year with low fee drag and works in both bull/bear markets with BTC/ETH edge.
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
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 for higher-timeframe trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator components (12h)
    # Jaw: EMA13, Teeth: EMA8, Lips: EMA5
    close_series = pd.Series(close)
    ema_5 = close_series.ewm(span=5, adjust=False, min_periods=5).mean().values   # Lips (fastest)
    ema_8 = close_series.ewm(span=8, adjust=False, min_periods=8).mean().values    # Teeth
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values # Jaw (slowest)
    
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
    
    # Warmup: max of EMA13 (13), ADX (14*2=28 for smoothing), 1w EMA34 alignment
    # 1w -> 12h: ~14 bars per week (7d * 2 = 14)
    start_idx = max(13, 28) + 14  # +14 to ensure 1w bar completion
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(ema_5[i]) or np.isnan(ema_8[i]) or np.isnan(ema_13[i]) or
            np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        lips = ema_5[i]
        teeth = ema_8[i]
        jaw = ema_13[i]
        adx_val = adx[i]
        pdi = plus_di[i]
        mdi = minus_di[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        close_val = close[i]
        
        # Regime conditions: ADX > 25 for trending market
        strong_trend = adx_val > 25
        
        # Williams Alligator alignment
        bullish_aligned = (lips > teeth) and (teeth > jaw)  # Lips > Teeth > Jaw
        bearish_aligned = (lips < teeth) and (teeth < jaw)  # Lips < Teeth < Jaw
        
        # Entry conditions
        long_entry = bullish_aligned and strong_trend and (close_val > ema_34_1w_val)
        short_entry = bearish_aligned and strong_trend and (close_val < ema_34_1w_val)
        
        # Exit conditions: reverse alignment or trend deterioration
        long_exit = not bullish_aligned or (adx_val < 20) or (close_val < ema_34_1w_val)
        short_exit = not bearish_aligned or (adx_val < 20) or (close_val > ema_34_1w_val)
        
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

name = "12h_WilliamsAlligator_Trend_Regime_ADX"
timeframe = "12h"
leverage = 1.0