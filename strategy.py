#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation (1.5x MA20).
# Enters long when price breaks above Camarilla R3 level with 12h bullish trend and volume > 1.5x MA20.
# Enters short when price breaks below Camarilla S3 level with 12h bearish trend and volume > 1.5x MA20.
# Exits when price crosses the Camarilla pivot point (mean reversion).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring strict confluence.
# Works in both bull and bear markets: 12h trend filter ensures alignment with higher timeframe direction,
# while Camarilla breakouts capture strong momentum moves and volume confirmation reduces false signals.
# Camarilla levels are derived from the previous completed 4h bar to avoid look-ahead.

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h close
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla pivot levels from previous completed 4h bar
    # Typical price for pivot calculation: (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Use previous bar's typical price to avoid look-ahead
    prev_typical = pd.Series(typical_price).shift(1).values
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Calculate pivot point (PP) from previous bar
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate Camarilla levels
    # R3 = PP + (High - Low) * 1.1 / 4
    # S3 = PP - (High - Low) * 1.1 / 4
    camarilla_r3 = pp + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = pp - (prev_high - prev_low) * 1.1 / 4
    camarilla_pp = pp  # Pivot point for exit
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_12h_aligned[i]) or \
           np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_pp[i]) or \
           np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 12h bullish trend and volume spike
            if close[i] > camarilla_r3[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with 12h bearish trend and volume spike
            elif close[i] < camarilla_s3[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Camarilla pivot point (mean reversion)
            if close[i] < camarilla_pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Camarilla pivot point (mean reversion)
            if close[i] > camarilla_pp[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals