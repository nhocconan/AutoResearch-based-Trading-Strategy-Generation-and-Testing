#!/usr/bin/env python3
"""
6h_WilliamsFractal_DailyTrend_WeeklyBias_v1
Hypothesis: Trade 6h Williams fractal breakouts with daily EMA34 trend filter and weekly EMA89 bias.
Long when: bullish fractal break + price > daily EMA34 + weekly EMA89 rising.
Short when: bearish fractal break + price < daily EMA34 + weekly EMA89 falling.
Uses volume confirmation (1.5x median) to avoid false breaks.
Position size 0.25 with ATR(14) trailing stop (2.0x).
Designed to work in both bull/bear via HTF trend/weekly bias filters.
Target: ~75-125 trades over 4 years (19-31/year) to minimize fee drag.
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
    
    # Get daily data for HTF trend and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly EMA(89) for bias filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_89_1w = pd.Series(df_1w['close'].values).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Williams fractals on daily (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Fractals need +2 daily bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # ATR for stop (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 1.5x median volume
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of daily EMA (34), weekly EMA (89), volume median (30), ATR (14)
    start_idx = max(34, 89, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_89_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        ema_89_1w_val = ema_89_1w_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: bullish fractal break + uptrend (close > daily EMA34) + weekly bias up (weekly EMA rising) + volume
            weekly_rising = ema_89_1w_val > ema_89_1w_aligned[i-1] if i > 0 else True
            long_signal = (bullish_fractal_val > 0) and \
                          (close_val > ema_34_1d_val) and \
                          weekly_rising and \
                          (volume_val > 1.5 * vol_median_val)
            # Short: bearish fractal break + downtrend (close < daily EMA34) + weekly bias down (weekly EMA falling) + volume
            weekly_falling = ema_89_1w_val < ema_89_1w_aligned[i-1] if i > 0 else False
            short_signal = (bearish_fractal_val > 0) and \
                           (close_val < ema_34_1d_val) and \
                           weekly_falling and \
                           (volume_val > 1.5 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < daily EMA34) after minimum holding period
            if bars_since_entry >= 4 and ((low_val < long_stop) or (close_val < ema_34_1d_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > daily EMA34) after minimum holding period
            if bars_since_entry >= 4 and ((high_val > short_stop) or (close_val > ema_34_1d_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_DailyTrend_WeeklyBias_v1"
timeframe = "6h"
leverage = 1.0