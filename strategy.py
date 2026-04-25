#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp_V7
Hypothesis: Add a choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) to avoid false breakouts in sideways markets. In trending regimes (CHOP < 38.2), trade Camarilla R1/S1 breakouts with 1d EMA34 trend and volume confirmation. In ranging regimes (CHOP > 61.8), fade the breakouts (mean reversion). This adaptive approach should work in both bull and bear markets by adjusting to market conditions. Target 20-40 trades/year to minimize fee drag.
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
    
    # 1d data for Camarilla pivots, EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    R1 = prev_close + 0.5 * prev_range
    S1 = prev_close - 0.5 * prev_range
    
    # Align 1d pivot levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness Index on 4h data (using 14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / (max_high - min_low)) / np.log10(14)
    chop[np.isnan(chop)] = 50  # neutral when undefined
    
    chop_trending = chop < 38.2   # trending regime
    chop_ranging = chop > 61.8    # ranging regime
    chop_neutral = ~(chop_trending | chop_ranging)  # neutral regime
    
    # ATR for stoploss (using 4h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d indicators (34 for EMA, 20 for vol MA) and 14 for chop
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals based on regime
            if chop_trending[i]:
                # Trending regime: trade breakouts with trend
                long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i]
                short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i]
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif short_breakout:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            elif chop_ranging[i]:
                # Ranging regime: fade breakouts (mean reversion)
                long_fade = (curr_close < S1_aligned[i]) and volume_spike[i]  # buy at support
                short_fade = (curr_close > R1_aligned[i]) and volume_spike[i]  # sell at resistance
                if long_fade:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif short_fade:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            # In neutral regime, no entries (wait for clearer signal)
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.0 * ATR below entry
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit based on regime
            elif chop_trending[i]:
                # In trending regime: exit if trend changes or price breaks below S1
                if not uptrend or curr_close < S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop_ranging[i]:
                # In ranging regime: exit if price reaches opposite extreme or volatility drops
                if curr_close > R1_aligned[i] or chop[i] < 38.2:  # reached resistance or regime changed
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Neutral regime: simple exit
                if curr_close < S1_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.0 * ATR above entry
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit based on regime
            elif chop_trending[i]:
                # In trending regime: exit if trend changes or price breaks above R1
                if not downtrend or curr_close > R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop_ranging[i]:
                # In ranging regime: exit if price reaches opposite extreme or volatility drops
                if curr_close < S1_aligned[i] or chop[i] < 38.2:  # reached support or regime changed
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Neutral regime: simple exit
                if curr_close > R1_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp_V7"
timeframe = "4h"
leverage = 1.0