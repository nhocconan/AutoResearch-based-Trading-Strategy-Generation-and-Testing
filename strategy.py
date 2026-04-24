#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R3 AND close > 1d EMA34 (bullish trend) AND volume > 1.5 * volume MA(20)
- Short when price breaks below Camarilla S3 AND close < 1d EMA34 (bearish trend) AND volume > 1.5 * volume MA(20)
- Exit on opposite Camarilla breakout (R3/S3) or trend reversal (close crosses 1d EMA34)
- Uses 4h primary timeframe with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Camarilla levels provide precise intraday support/resistance that works in ranging markets
- 1d EMA34 ensures alignment with long-term trend to avoid whipsaws in bear markets
- Volume spike confirmation filters low-momentum breakouts
- Designed for BTC/ETH edge in both bull (breakout continuation) and bear (trend-aligned retracements) markets
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
    
    # Calculate Camarilla levels (R3, S3) using previous period (no look-ahead)
    lookback = 20
    # Typical price for Camarilla calculation
    typical_price = (high + low + close) / 3.0
    # Pivot point = typical price of previous period
    pp = pd.Series(typical_price).shift(1).values
    # Range = high - low of previous period
    range_hl = (pd.Series(high).shift(1) - pd.Series(low).shift(1)).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = pp + range_hl * 1.1 / 4.0
    camarilla_s3 = pp - range_hl * 1.1 / 4.0
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, trend up (close > EMA34), volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, trend down (close < EMA34), volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 OR trend reversal (close < EMA34)
            if close[i] < camarilla_s3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 OR trend reversal (close > EMA34)
            if close[i] > camarilla_r3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0