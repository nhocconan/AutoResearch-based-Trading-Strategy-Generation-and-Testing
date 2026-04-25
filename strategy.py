#!/usr/bin/env python3
"""
6h_ADX_DMI_ElderRay_Regime
Hypothesis: On 6h timeframe, combine ADX/DMI trend strength with Elder Ray (Bull/Bear Power) to filter entries.
In bull markets: ADX > 25 + +DI > -DI + Bull Power > 0 → long
In bear markets: ADX > 25 + -DI > +DI + Bear Power < 0 → short
In ranging markets (ADX < 20): fade extremes using Bollinger Bands (20,2) with volume confirmation.
Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Discrete position sizing (0.25) to limit fee drag. Targets 15-30 trades/year.
Works in trending markets via ADX/DMI/Elder Ray and in ranging markets via mean reversion.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for higher timeframe trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX/DMI calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - low[:-1])
    tr3 = np.abs(low[1:] - high[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, 0)
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Bollinger Bands for ranging regime (20,2)
    bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX (14*3), EMA13 (13), BB (20), volume MA (20), 1d EMA50 (50)
    start_idx = max(50, 20, 14*3, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending regime: ADX > 25
            if adx[i] > 25:
                # Long: +DI > -DI + Bull Power > 0 + 1d uptrend + volume confirmation
                if (plus_di[i] > minus_di[i]) and (bull_power[i] > 0) and \
                   (close[i] > ema_50_1d_aligned[i]) and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: -DI > +DI + Bear Power < 0 + 1d downtrend + volume confirmation
                elif (minus_di[i] > plus_di[i]) and (bear_power[i] < 0) and \
                     (close[i] < ema_50_1d_aligned[i]) and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging regime: ADX < 20
            elif adx[i] < 20:
                # Long: price at lower BB + volume spike
                if (close[i] <= bb_lower[i]) and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price at upper BB + volume spike
                elif (close[i] >= bb_upper[i]) and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Hold long position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Trend reversal: ADX > 25 and -DI > +DI
            # 2. Bear power turns negative in trending market
            # 3. Price reaches opposite BB in ranging market
            # 4. Volume spike with reversal signal
            exit = False
            if adx[i] > 25:
                if (minus_di[i] > plus_di[i]) or (bear_power[i] < 0):
                    exit = True
            else:  # ranging
                if close[i] >= bb_ma[i]:  # exit at middle BB
                    exit = True
            # Additional exit: volume spike with opposite signal
            if volume_spike[i]:
                if adx[i] > 25 and bear_power[i] < 0:
                    exit = True
                elif adx[i] < 20 and close[i] >= bb_ma[i]:
                    exit = True
            if exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Trend reversal: ADX > 25 and +DI > -DI
            # 2. Bull power turns positive in trending market
            # 3. Price reaches opposite BB in ranging market
            # 4. Volume spike with reversal signal
            exit = False
            if adx[i] > 25:
                if (plus_di[i] > minus_di[i]) or (bull_power[i] > 0):
                    exit = True
            else:  # ranging
                if close[i] <= bb_ma[i]:  # exit at middle BB
                    exit = True
            # Additional exit: volume spike with opposite signal
            if volume_spike[i]:
                if adx[i] > 25 and bull_power[i] > 0:
                    exit = True
                elif adx[i] < 20 and close[i] <= bb_ma[i]:
                    exit = True
            if exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_DMI_ElderRay_Regime"
timeframe = "6h"
leverage = 1.0