#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume spike + 12h chop regime filter
    # Donchian breakout provides directional edge in trends
    # Volume spike confirms institutional participation
    # 12h chop filter avoids false breakouts in ranging markets
    # Works in bull/bear by only taking breakouts in choppy regimes (mean reversion) 
    # and avoiding breakouts in trending regimes (which often fail in bear markets)
    # Target: 20-50 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for chop regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h ATR14 for chop calculation
    tr_12h = np.zeros(len(df_12h))
    for i in range(len(df_12h)):
        if i == 0:
            tr_12h[i] = high_12h[i] - low_12h[i]
        else:
            tr_12h[i] = max(
                high_12h[i] - low_12h[i],
                abs(high_12h[i] - close_12h[i-1]),
                abs(low_12h[i] - close_12h[i-1])
            )
    
    atr14_12h = np.full(len(df_12h), np.nan)
    for i in range(13, len(df_12h)):
        if i == 13:
            atr14_12h[i] = np.mean(tr_12h[i-13:i+1])
        else:
            atr14_12h[i] = (atr14_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate 12h True Range sum and ATR sum for chopiness
    tr_sum_14 = np.full(len(df_12h), np.nan)
    atr_sum_14 = np.full(len(df_12h), np.nan)
    for i in range(13, len(df_12h)):
        if i == 13:
            tr_sum_14[i] = np.sum(tr_12h[i-13:i+1])
            atr_sum_14[i] = np.sum(atr14_12h[i-13:i+1])
        else:
            tr_sum_14[i] = tr_sum_14[i-1] - tr_12h[i-14] + tr_12h[i]
            atr_sum_14[i] = atr_sum_14[i-1] - atr14_12h[i-14] + atr14_12h[i]
    
    # Chopiness Index: CHOP = 100 * log10(atr_sum_14 / tr_sum_14) / log10(14)
    chop_12h = np.full(len(df_12h), np.nan)
    for i in range(13, len(df_12h)):
        if tr_sum_14[i] > 0 and atr_sum_14[i] > 0:
            chop_12h[i] = 100 * np.log10(atr_sum_14[i] / tr_sum_14[i]) / np.log10(14)
        else:
            chop_12h[i] = 50.0  # neutral when undefined
    
    # Align 12h chop to 4h timeframe
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian(20) channels
    highest_20 = np.full(len(df_4h), np.nan)
    lowest_20 = np.full(len(df_4h), np.nan)
    for i in range(19, len(df_4h)):
        highest_20[i] = np.max(high_4h[i-19:i+1])
        lowest_20[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 4h volume MA(20) for volume spike filter
    vol_ma_20 = np.full(len(df_4h), np.nan)
    for i in range(19, len(df_4h)):
        if i == 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        else:
            vol_ma_20[i] = (vol_ma_20[i-1] * 19 + volume_4h[i]) / 20
    
    # Align 4h indicators to LTF
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x 20-period average
        volume_spike = volume[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # Chop regime filter: trade only when CHOP > 50 (ranging/mean reverting)
        # In choppy markets, breakouts often fail quickly, so we fade them
        # In trending markets (CHOP < 50), breakouts tend to continue
        chop_value = chop_12h_aligned[i]
        in_chop_regime = chop_value > 50
        in_trend_regime = chop_value <= 50
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_20_aligned[i]
        breakout_down = close[i] < lowest_20_aligned[i]
        
        # Entry logic:
        # In choppy regimes: fade breakouts (mean reversion)
        # In trending regimes: follow breakouts (trend continuation)
        long_entry = False
        short_entry = False
        
        if in_chop_regime and volume_spike:
            # Fade breakouts in choppy markets
            if breakout_down:
                long_entry = True  # Expect reversion up from lower band
            elif breakout_up:
                short_entry = True  # Expect reversion down from upper band
        elif in_trend_regime and volume_spike:
            # Follow breakouts in trending markets
            if breakout_up:
                long_entry = True  # Expect continuation up
            elif breakout_down:
                short_entry = True  # Expect continuation down
        
        # Exit when price returns to mid-channel or opposite breakout
        mid_channel = (highest_20_aligned[i] + lowest_20_aligned[i]) / 2
        long_exit = position == 1 and (close[i] <= mid_channel or breakout_down)
        short_exit = position == -1 and (close[i] >= mid_channel or breakout_up)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_chop_vol_v1"
timeframe = "4h"
leverage = 1.0