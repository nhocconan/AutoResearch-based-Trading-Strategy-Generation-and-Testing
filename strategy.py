#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_Volume_ChopRegime
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume confirmation, and choppiness regime filter.
Only trades when market is trending (CHOP < 38.2) to avoid whipsaws in ranging markets.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 20-40 trades/year on BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34 trend filter and Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (more conservative than H3/L3)
    R1 = prev_close + 1.1 * prev_range * (1.0/12.0)  # R1 = C + 1.1*(HL/12)
    S1 = prev_close - 1.1 * prev_range * (1.0/12.0)  # S1 = C - 1.1*(HL/12)
    
    # Align 1d levels to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d choppiness regime filter (CHOP < 38.2 = trending market)
    chop_window = 14
    true_range = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
            np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)
        )
    )
    atr_1d = pd.Series(true_range).rolling(window=chop_window, min_periods=chop_window).mean().values
    highest_high = pd.Series(df_1d['high'].values).rolling(window=chop_window, min_periods=chop_window).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=chop_window, min_periods=chop_window).min().values
    chop = 100 * np.log10(atr_1d * chop_window / (highest_high - lowest_low)) / np.log10(chop_window)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 1.8 * 20-period average (on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and choppiness (14)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            # Look for entry signals with volume spike, trend alignment, and trending regime
            # Long breakout: price breaks above R1 with uptrend, volume spike, and trending regime
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i] and trending_regime
            # Short breakout: price breaks below S1 with downtrend, volume spike, and trending regime
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i] and trending_regime
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.0 * ATR below entry (using 1d ATR aligned to 4h)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below S1 (mean reversion) or trend changes or regime shifts to choppy
            elif curr_close < S1_aligned[i] or not uptrend or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Calculate 1d ATR aligned to 4h (same as above)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above R1 (mean reversion) or trend changes or regime shifts to choppy
            elif curr_close > R1_aligned[i] or not downtrend or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_ChopRegime"
timeframe = "4h"
leverage = 1.0