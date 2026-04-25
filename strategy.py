#!/usr/bin/env python3
"""
12h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Williams fractals on 12h identify major swing points with 2-bar confirmation delay. 
Breakouts above/below these levels with 1d EMA34 trend filter, volume spike, and chop regime filter capture strong momentum moves while avoiding whipsaws in ranging markets. 
Works in bull (buy breakouts above bearish fractals in uptrend) and bear (sell breakdowns below bullish fractals in downtrend) via symmetric logic. 
Target 12-37 trades/year on 12h to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for choppiness index (regime filter)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for chop
    tr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(df_1d['high'].iloc[i] - df_1d['low'].iloc[i], 
                       abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]), 
                       abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1]))
    atr_1d = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Calculate 1d Chop = 100 * log15(sum(ATR14) / (max(high14)-min(low14)))
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        sum_atr = np.sum(atr_1d[i-13:i+1])
        max_high = np.max(df_1d['high'].iloc[i-13:i+1])
        min_low = np.min(df_1d['low'].iloc[i-13:i+1])
        if max_high > min_low:
            chop_1d[i] = 100 * np.log14(sum_atr / (max_high - min_low)) / np.log14(15)
        else:
            chop_1d[i] = 50.0  # neutral if no range
    
    # Align chop with 1d close (no extra delay needed for chop)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 12h data for Williams fractals (need 2 extra bars for confirmation)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 12h
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_12h['high'].values,
        df_12h['low'].values
    )
    # Align with 2 extra delay bars for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # Calculate ATR(14) for stop management on 12h timeframe
    atr_12h = np.full(n, np.nan)
    tr_12h = np.zeros(n)
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_12h[i] = np.mean(tr_12h[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation on 12h
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA34, ATR, volume MA, fractals, chop
    start_idx = max(34, 14, 20, 5)  # 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        atr_val = atr_12h[i]
        vol_ma = vol_ma_20[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Regime filter: chop < 61.8 = trending (good for breakouts), chop > 61.8 = ranging (avoid breakouts)
        trending_regime = chop_val < 61.8
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at fractal levels
            # Long: price breaks above bearish fractal with volume confirmation in uptrend AND trending regime
            long_breakout = (curr_close > bearish_fractal_val) and volume_confirm and uptrend and trending_regime
            # Short: price breaks below bullish fractal with volume confirmation in downtrend AND trending regime
            short_breakout = (curr_close < bullish_fractal_val) and volume_confirm and downtrend and trending_regime
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below bullish fractal OR 2.5*ATR trailing stop OR EMA34 trend turns down OR chop becomes too high (range)
            if (curr_close < bullish_fractal_val or 
                curr_close < (highest_since_entry - 2.5 * atr_val) or 
                curr_close < ema_34_val or 
                chop_val > 61.8):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above bearish fractal OR 2.5*ATR trailing stop OR EMA34 trend turns up OR chop becomes too high (range)
            if (curr_close > bearish_fractal_val or 
                curr_close > (lowest_since_entry + 2.5 * atr_val) or 
                curr_close > ema_34_val or 
                chop_val > 61.8):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0