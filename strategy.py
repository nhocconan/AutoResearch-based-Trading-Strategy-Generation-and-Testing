#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_RegimeFilter
Hypothesis: On 4h timeframe, trade Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume spike confirmation. Uses Bollinger Bandwidth percentile regime filter to avoid false breakouts in choppy markets. Designed for low trade frequency (20-50/year) to minimize fee drift. Works in bull/bear via trend filter and regime adaptation.
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
    
    # 1d data for Camarilla pivots, EMA34, Bollinger Bandwidth (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H3 = prev_close + 1.125 * prev_range
    L3 = prev_close - 1.125 * prev_range
    H4 = prev_close + 1.5 * prev_range
    L4 = prev_close - 1.5 * prev_range
    R1 = prev_close + prev_range * 0.125
    S1 = prev_close - prev_range * 0.125
    
    # Align 1d pivot levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
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
    
    # Volume spike: current volume > 2.0 * 20-period average (tighter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for stoploss (using 4h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d indicators (50 for BBWP percentile, 20 for vol MA, 34 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Regime determination based on Bollinger Bandwidth percentile
        bbwp = bb_width_percentile_aligned[i]
        low_vol_regime = bbwp < 30   # Ranging market
        high_vol_regime = bbwp > 70  # Trending market
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - tighter conditions
            # Long: break above R1 in uptrend + high vol + volume spike
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and high_vol_regime and volume_spike[i]
            # Short: break below S1 in downtrend + high vol + volume spike
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and high_vol_regime and volume_spike[i]
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.0 * ATR below entry
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: reach H3 or H4
            elif curr_close >= H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Regime change to low vol - exit to avoid whipsaw
            elif low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.0 * ATR above entry
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: reach L3 or L4
            elif curr_close <= L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Regime change to low vol - exit to avoid whipsaw
            elif low_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0