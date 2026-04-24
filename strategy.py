#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 100-180 total trades over 4 years (25-45/year).
- HTF: 12h for EMA50 trend direction.
- Donchian Channel: 20-period high/low breakouts capture volatility expansion.
- Entry: Long when price breaks above 20-period high AND price > 12h EMA50 AND volume > 1.5 * 20-period average volume.
         Short when price breaks below 20-period low AND price < 12h EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout or EMA trend flip.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull/bear: Donchian breakouts capture momentum moves; EMA filter ensures trend alignment; volume confirms legitimacy.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian Channel (20-period)
    donchian_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian(20) and EMA50 need 50 bars
    
    for i in range(start_idx, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        curr_vol_ma = vol_ma_20[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below Donchian low OR EMA trend turns bearish
            if position == 1:
                if curr_low <= donchian_low[i] or curr_close < ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR EMA trend turns bullish
            elif position == -1:
                if curr_high >= donchian_high[i] or curr_close > ema_50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions
        if position == 0:
            # Donchian breakout signals
            breakout_up = curr_high >= donchian_high[i] and prev_close < donchian_high[i-1]
            breakout_down = curr_low <= donchian_low[i] and prev_close > donchian_low[i-1]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * curr_vol_ma if not np.isnan(curr_vol_ma) else False
            
            # EMA trend filter: price above/below 12h EMA50
            price_above_ema = curr_close > ema_50_12h_aligned[i]
            price_below_ema = curr_close < ema_50_12h_aligned[i]
            
            if breakout_up and volume_confirm and price_above_ema:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0