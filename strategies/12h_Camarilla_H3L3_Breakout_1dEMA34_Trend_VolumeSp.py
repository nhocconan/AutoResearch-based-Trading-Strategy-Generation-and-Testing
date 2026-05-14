#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSp
Hypothesis: On 12h timeframe, use Camarilla H3/L3 levels for mean reversion in ranging markets and H4/L4 for breakout in trending markets, with 1-day EMA34 as trend filter and volume spike confirmation. Designed for low trade frequency (12-37/year) to minimize fee drag. Works in both bull and bear markets by adapting to volatility regimes via Bollinger Bandwidth percentile. Includes ATR-based stoploss to control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots, Bollinger Bandwidth, and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H4 = prev_close + 1.5 * prev_range
    L4 = prev_close - 1.5 * prev_range
    H3 = prev_close + 1.125 * prev_range
    L3 = prev_close - 1.125 * prev_range
    
    # Align 1d pivot levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1d Bollinger Bandwidth for regime filter (20, 2)
    bb_middle = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr = None
    
    # Calculate ATR for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Start index: need enough for 1d indicators (50 for BBWP percentile, 20 for vol MA, 34 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Regime determination based on Bollinger Bandwidth percentile
        bbwp = bb_width_percentile_aligned[i]
        low_vol_regime = bbwp < 30   # Ranging market: mean revert at H3/L3
        high_vol_regime = bbwp > 70  # Trending market: breakout at H4/L4
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long signals
            long_mean_revert = (curr_close <= L3_aligned[i]) and low_vol_regime
            long_breakout = (curr_close >= H4_aligned[i]) and high_vol_regime and uptrend
            
            # Short signals
            short_mean_revert = (curr_close >= H3_aligned[i]) and low_vol_regime
            short_breakout = (curr_close <= L4_aligned[i]) and high_vol_regime and downtrend
            
            long_entry = (long_mean_revert or long_breakout) and volume_spike[i]
            short_entry = (short_mean_revert or short_breakout) and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.5 * ATR below entry
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit in low vol
            elif low_vol_regime and curr_close >= H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Breakdown exit in high vol
            elif high_vol_regime and curr_close <= L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Regime change against position
            elif low_vol_regime and not (curr_close <= L3_aligned[i] or curr_close >= H3_aligned[i]):
                # Still in range, hold
                signals[i] = 0.25
            elif high_vol_regime and not (curr_close >= H4_aligned[i] or curr_close <= L4_aligned[i]):
                # Still in trend, hold
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.5 * ATR above entry
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit in low vol
            elif low_vol_regime and curr_close <= L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Breakout exit in high vol
            elif high_vol_regime and curr_close >= H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Regime change against position
            elif low_vol_regime and not (curr_close <= L3_aligned[i] or curr_close >= H3_aligned[i]):
                # Still in range, hold
                signals[i] = -0.25
            elif high_vol_regime and not (curr_close >= H4_aligned[i] or curr_close <= L4_aligned[i]):
                # Still in trend, hold
                signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSp"
timeframe = "12h"
leverage = 1.0