#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R3 AND close > 1d EMA34 (bullish trend) AND volume > 1.5 * volume MA20
- Short when price breaks below Camarilla S3 AND close < 1d EMA34 (bearish trend) AND volume > 1.5 * volume MA20
- Exit on opposite Camarilla breakout (R4/S4) or trend reversal (close crosses 1d EMA34)
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Camarilla levels provide precise intraday support/resistance that works in ranging markets
- 1d EMA34 ensures alignment with medium-term trend to avoid whipsaws and capture major moves
- Volume spike filter confirms institutional participation, reducing fakeouts
- Designed for BTC/ETH edge: works in bull markets (breakout continuation) and bear markets (avoiding false breakouts via trend filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 4h timeframe using previous period (no look-ahead)
    lookback = 20
    rolling_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1)
    rolling_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1)
    rolling_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1)
    
    # Camarilla calculations
    rang = rolling_high - rolling_low
    camarilla_h5 = rolling_close + rang * 1.1 / 2
    camarilla_h4 = rolling_close + rang * 1.1 / 4
    camarilla_h3 = rolling_close + rang * 1.1 / 6
    camarilla_l3 = rolling_close - rang * 1.1 / 6
    camarilla_l4 = rolling_close - rang * 1.1 / 4
    camarilla_l5 = rolling_close - rang * 1.1 / 2
    
    camarilla_h3 = camarilla_h3.values
    camarilla_l3 = camarilla_l3.values
    camarilla_h4 = camarilla_h4.values
    camarilla_l4 = camarilla_l4.values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3, trend up (close > EMA34), volume confirmation
            if close[i] > camarilla_h3[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3, trend down (close < EMA34), volume confirmation
            elif close[i] < camarilla_l3[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks above Camarilla H4 (take profit) OR below L3 (stop/reverse) OR trend reversal (close < EMA34)
            if close[i] > camarilla_h4[i] or close[i] < camarilla_l3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks below Camarilla L4 (take profit) OR above H3 (stop/reverse) OR trend reversal (close > EMA34)
            if close[i] < camarilla_l4[i] or close[i] > camarilla_h3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0