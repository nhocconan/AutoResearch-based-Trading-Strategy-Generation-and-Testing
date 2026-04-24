#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 Breakout + 1d ADX Trend Filter + Volume Confirmation.
- Primary timeframe: 12h for lower trade frequency and better signal quality.
- HTF: 1d for ADX regime filter (trending vs ranging).
- Camarilla levels calculated from prior 1d candle: H3/L3 act as breakout levels.
- In trending regime (ADX > 25): breakouts above H3 go long, below L3 go short.
- In ranging regime (ADX < 20): fade extremes at H3/L3 for mean reversion.
- Volume confirmation: current volume > 1.5x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to balance profit potential and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying breakouts in uptrends, in bear via selling breakdowns,
  and in range via mean reversion at extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Calculate 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    plus_di = 100 * plus_di_smooth / atr
    minus_di = 100 * minus_di_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate prior 1d Camarilla levels for breakout levels
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # Need ADX(14)+buffer, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime: ADX > 25 = trending, ADX < 20 = ranging
            if adx_aligned[i] > 25:
                # Trending regime: breakout trading
                if close[i] > camarilla_h3_aligned[i] and volume_spike[i]:
                    # Bullish breakout above H3
                    signals[i] = 0.25
                    position = 1
                elif close[i] < camarilla_l3_aligned[i] and volume_spike[i]:
                    # Bearish breakdown below L3
                    signals[i] = -0.25
                    position = -1
            elif adx_aligned[i] < 20:
                # Ranging regime: mean reversion at extremes
                if close[i] < camarilla_l3_aligned[i] and volume_spike[i]:
                    # Price below L3: buy mean reversion
                    signals[i] = 0.25
                    position = 1
                elif close[i] > camarilla_h3_aligned[i] and volume_spike[i]:
                    # Price above H3: sell mean reversion
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to Camarilla H4 level or reverse signal
            if close[i] < camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Camarilla L4 level or reverse signal
            if close[i] > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dADX_Regime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0