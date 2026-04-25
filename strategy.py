#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 12-hour Camarilla H3/L3 breakout with 1-day EMA34 trend filter, volume confirmation, and choppiness regime filter.
Targets 12-37 trades/year by requiring: 1) price breaks daily H3/L3 levels, 2) aligned with 1d EMA34 trend,
3) volume > 1.8x 20-period average, 4) choppy market filter (Choppiness Index > 61.8) for mean-reversion logic.
Uses 12h timeframe to minimize fee drag while capturing significant moves. Volume spike reduces false breakouts.
Chop filter ensures strategy only operates in ranging markets where Camarilla mean reversion works best.
Designed to work in both bull and bear markets by following the 1d trend direction for breakouts and
using mean reversion in choppy regimes.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Camarilla pivots and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla H3 and L3 levels (H3 = C + 1.1*(HL/2), L3 = C - 1.1*(HL/2))
    H3 = prev_close + 1.1 * prev_range * (1.0/2.0)
    L3 = prev_close - 1.1 * prev_range * (1.0/2.0)
    
    # Align 1d levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    # Choppiness Index regime filter (loaded ONCE for 1d)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n*(highest-high - lowest-low)))
    # Simplified: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
                                  np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    highest_14_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_14_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    atr_sum = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    price_range = highest_14_1d - lowest_14_1d
    chop_raw = 100 * np.log10(atr_sum / np.log10(14)) / np.log10(price_range)
    chop_raw = np.where(price_range > 0, chop_raw, 50.0)  # neutral when range=0
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + EMA34 (34) + volume MA (20) + ATR (14) + HH/LL (14)
    start_idx = 34 + 20 + 14 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Regime filter: choppy market (CHOP > 61.8) for mean reversion
        choppy_market = chop_aligned[i] > 61.8
        
        if position == 0:
            # Look for entry signals with volume confirmation and regime logic
            # In choppy markets: mean reversion at H3/L3 (fade extreme moves)
            # In trending markets: breakout continuation
            
            if choppy_market:
                # Mean reversion logic in ranging markets
                # Long when price rejects L3 and shows bullish rejection
                long_setup = (curr_low <= L3_aligned[i] * 1.002) and (curr_close > L3_aligned[i]) and uptrend
                # Short when price rejects H3 and shows bearish rejection
                short_setup = (curr_high >= H3_aligned[i] * 0.998) and (curr_close < H3_aligned[i]) and downtrend
                
                if long_setup and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif short_setup and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                else:
                    signals[i] = 0.0
            else:
                # Breakout logic in trending markets
                # Long breakout: price breaks above H3 with uptrend and volume confirmation
                long_breakout = (curr_close > H3_aligned[i]) and uptrend and volume_confirm[i]
                # Short breakout: price breaks below L3 with downtrend and volume confirmation
                short_breakout = (curr_close < L3_aligned[i]) and downtrend and volume_confirm[i]
                
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
            # Exit if price reaches opposite level (mean reversion) or trend/chop regime changes
            if choppy_market:
                # In choppy: mean reversion to mid-point or opposite level
                mid_point = (H3_aligned[i] + L3_aligned[i]) / 2
                if curr_close >= mid_point or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In trending: trail with trend or break below L3
                if curr_close < L3_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price reaches opposite level (mean reversion) or trend/chop regime changes
            if choppy_market:
                # In choppy: mean reversion to mid-point or opposite level
                mid_point = (H3_aligned[i] + L3_aligned[i]) / 2
                if curr_close <= mid_point or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In trending: trail with trend or break above H3
                if curr_close > H3_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0