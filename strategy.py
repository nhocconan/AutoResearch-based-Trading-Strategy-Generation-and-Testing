#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_ChopFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h timeframe with 1d EMA34 trend filter and chop regime filter.
In trending markets (ADX>25): buy breakouts above R1 in uptrend, sell breakdowns below S1 in downtrend.
In ranging markets (ADX<20): fade touches of R1/S1 with mean reversion.
Uses 1d EMA34 for HTF trend and 12h ADX for regime detection.
Position size: 0.25 to limit drawdown. Target: 15-25 trades/year.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h ADX for regime filter (trending vs ranging)
    # ADX calculation: +DM, -DM, TR, then smoothed
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    atr[0] = tr[0]
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        atr[i] = atr[i-1] + alpha * (tr[i] - atr[i-1])
        plus_dm_smooth[i] = plus_dm_smooth[i-1] + alpha * (plus_dm[i] - plus_dm_smooth[i-1])
        minus_dm_smooth[i] = minus_dm_smooth[i-1] + alpha * (minus_dm[i] - minus_dm_smooth[i-1])
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = np.zeros_like(dx)
    adx[period-1] = dx[period-1]
    for i in range(period, len(dx)):
        adx[i] = adx[i-1] + alpha * (dx[i] - adx[i-1])
    
    # Calculate 12h Camarilla pivot levels (R1, S1) from previous day
    # For 12h timeframe, we need to calculate pivots from daily OHLC
    # We'll use the 1d data to calculate Camarilla levels for each 12h bar
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # But we need to align these to 12h bars
    
    # Get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (each day's levels apply to 2x 12h bars)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX (34+14=48) and EMA34 (34)
    start_idx = 48
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        # Determine 1d HTF trend (bullish = price above EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            if is_trending:
                # Trending regime: breakout strategy
                long_setup = (close[i] > camarilla_r1_aligned[i]) and htf_1d_bullish
                short_setup = (close[i] < camarilla_s1_aligned[i]) and htf_1d_bearish
                
                if long_setup:
                    signals[i] = 0.25
                    position = 1
                elif short_setup:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging regime: mean reversion at pivot levels
                long_setup = (close[i] < camarilla_s1_aligned[i]) and (close[i] > camarilla_s1_aligned[i] * 0.998)
                short_setup = (close[i] > camarilla_r1_aligned[i]) and (close[i] < camarilla_r1_aligned[i] * 1.002)
                
                if long_setup:
                    signals[i] = 0.25
                    position = 1
                elif short_setup:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX 20-25): no trades
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if is_trending:
                # Exit on trend reversal or touch of S1 (stop)
                if (not htf_1d_bullish) or (close[i] <= camarilla_s1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
            else:
                # Ranging: exit at midpoint or opposite level
                midpoint = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2
                if close[i] >= midpoint:
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if is_trending:
                # Exit on trend reversal or touch of R1 (stop)
                if htf_1d_bullish or (close[i] >= camarilla_r1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
            else:
                # Ranging: exit at midpoint or opposite level
                midpoint = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2
                if close[i] <= midpoint:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0