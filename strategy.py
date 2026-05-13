#!/usr/bin/env python3
"""
1d_Alligator_ElderRay_Trend_Filter
Hypothesis: Williams Alligator (13,8,5 SMAs) + Elder Ray (bull/bear power) on 1d with 1w trend filter works in both bull and bear markets. 
Long when green line > red line (bullish alignment) and bull power > 0 with 1w uptrend. 
Short when red line > green line (bearish alignment) and bear power < 0 with 1w downtrend.
Exit on opposite Alligator alignment or trend filter failure. Uses volume confirmation to avoid whipsaws.
Target: 10-25 trades/year per symbol.
"""

name = "1d_Alligator_ElderRay_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMA(13), SMA(8), SMA(5) on median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # Green line
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Alligator alignment: Green > Red > Blue = bullish, Red > Green > Blue = bearish
    bullish_align = (lips > teeth) & (teeth > jaw)
    bearish_align = (teeth > lips) & (lips > jaw)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1w = df_1w['close'].values > ema_20_1w
    downtrend_1w = df_1w['close'].values < ema_20_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        bull_align = bullish_align[i]
        bear_align = bearish_align[i]
        bull_pow = bull_power[i]
        bear_pow = bear_power[i]
        vol_conf = volume_conf[i]
        uptrend_htf = uptrend_1w_aligned[i]
        downtrend_htf = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: bullish Alligator alignment + bull power > 0 + 1w uptrend + volume confirmation
            if bull_align and (bull_pow > 0) and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish Alligator alignment + bear power < 0 + 1w downtrend + volume confirmation
            elif bear_align and (bear_pow < 0) and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish Alligator alignment or bull power <= 0 or 1w trend turns down
            if bear_align or (bull_pow <= 0) or not uptrend_htf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish Alligator alignment or bear power >= 0 or 1w trend turns up
            if bull_align or (bear_pow >= 0) or not downtrend_htf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals