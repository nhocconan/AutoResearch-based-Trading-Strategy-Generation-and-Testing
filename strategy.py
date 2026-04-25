#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime
Hypothesis: 12-hour Camarilla R1/S1 breakout with 1-day EMA34 trend filter, volume confirmation, and choppiness regime filter.
Targets 12-37 trades/year by requiring: 1) price breaks daily R1/S1 levels (key intraday support/resistance),
2) aligned with 1d EMA34 trend, 3) volume > 1.5x 20-period average, 4) choppiness index < 61.8 (trending market).
Uses 12h timeframe to minimize fee drag while capturing significant moves. R1/S1 levels provide
higher signal quality than H3/L3. Regime filter avoids whipsaws in ranging markets. Works in both bull and bear
by only taking trend-aligned breakouts with volume confirmation, reducing false signals.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d data for Camarilla pivots (loaded ONCE)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # Align 1d levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Choppiness regime filter: CHOP < 61.8 = trending market (use 1d data)
    # Calculate True Range and ATR(14) for 1d
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14)/ (n * ATR)) / log10(n)
    # where n = 14 periods
    sum_atr14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (np.log10(sum_atr14 / (14 * atr_14 + 1e-10)) / np.log10(14))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)
    
    # Regime filter: trending market (CHOP < 61.8)
    trending_regime = chop_aligned < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and previous day data (1) + chop calculation (14+14)
    start_idx = 34 + 14 + 14 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment, and regime filter
            # Long breakout: price breaks above R1 with uptrend, volume confirmation, and trending regime
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_confirm[i] and trending_regime[i]
            # Short breakout: price breaks below S1 with downtrend, volume confirmation, and trending regime
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_confirm[i] and trending_regime[i]
            
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
            # Exit if price breaks below S1 (mean reversion) or trend changes or regime changes to ranging
            if curr_close < S1_aligned[i] or not uptrend or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above R1 (mean reversion) or trend changes or regime changes to ranging
            if curr_close > R1_aligned[i] or not downtrend or not trending_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0