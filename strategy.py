#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_v1
Hypothesis: 12-hour Camarilla R1/S1 breakout with daily EMA34 trend filter and choppiness regime filter.
Designed for low trade frequency (target 12-37/year) to minimize fee drag while capturing medium-term swings.
The daily EMA34 provides strong trend filtering that works across bull/bear regimes, and breakouts at 
Camarilla R1/S1 levels offer precise entries. Choppiness filter avoids whipsaws in ranging markets.
Focuses on BTC and ETH as primary targets for robustness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on daily for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous daily bar
    # Camarilla: R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    # Align Camarilla levels to 12h (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate Choppiness Index on 12h for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # We'll use a simplified version: Chop > 61.8 = ranging, Chop < 38.2 = trending
    atr_period = 14
    chop_period = 14
    tr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(chop_period)
    chop = np.where(np.isnan(chop), 50.0, chop)  # default to middle if not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of daily EMA(34), ATR, Chop
    start_idx = max(34, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # Daily uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # Daily downtrend
        
        # Regime filter: only trade in trending markets (Chop < 38.2)
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND daily trend up AND trending regime
            long_signal = (close_val > camarilla_r1_aligned[i]) and trend_1d_up and is_trending
            
            # Short: price breaks below Camarilla S1 AND daily trend down AND trending regime
            short_signal = (close_val < camarilla_s1_aligned[i]) and trend_1d_down and is_trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend flips down OR price hits ATR stoploss OR regime changes to ranging
            if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]) or (not is_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss OR regime changes to ranging
            if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]) or (not is_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0