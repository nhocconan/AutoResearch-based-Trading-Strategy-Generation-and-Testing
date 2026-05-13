#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation. Uses 1w Camarilla pivot levels for directional bias. Designed for BTC/ETH robustness: Bollinger squeeze identifies low volatility primed for breakout, ADX>25 confirms trending regime, volume spike validates breakout strength, and 1w Camarilla pivot (R1/S1) provides institutional reference points for continuation. Targets 12-37 trades/year on 6h timeframe.

name = "6h_BBandSqueeze_Breakout_1dADX_VolumeSpike_1wCamarilla_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + bb_std * bb_stddev
    lower_band = sma - bb_std * bb_stddev
    bb_width = (upper_band - lower_band) / sma  # normalized width
    
    # Bollinger Band Squeeze: width < 20-period average width * 0.8
    avg_bb_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < 0.8 * avg_bb_width
    
    # Breakout conditions: price > upper band (long) or < lower band (short)
    breakout_long = close > upper_band
    breakout_short = close < lower_band
    
    # Calculate 1d ADX for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trend_condition = adx_aligned > 25  # trending regime
    
    # Volume confirmation: current volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_condition = volume > 1.5 * avg_volume
    
    # Calculate 1w Camarilla pivot for directional bias (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    camarilla_r1 = close_1w + (high_1w - low_1w) * 1.1 / 12
    camarilla_s1 = close_1w - (high_1w - low_1w) * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Directional bias: price > weekly R1 = bullish bias, < weekly S1 = bearish bias
    bullish_bias = close > camarilla_r1_aligned
    bearish_bias = close < camarilla_s1_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(sma[i]) or np.isnan(bb_width[i]) or np.isnan(avg_bb_width[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: squeeze breakout up + bullish bias + volume + trend
            if (squeeze_condition[i] and breakout_long[i] and 
                bullish_bias[i] and volume_condition[i] and trend_condition[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: squeeze breakout down + bearish bias + volume + trend
            elif (squeeze_condition[i] and breakout_short[i] and 
                  bearish_bias[i] and volume_condition[i] and trend_condition[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes below middle band OR squeeze re-establishes
            if (close[i] < sma[i] or squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes above middle band OR squeeze re-establishes
            if (close[i] > sma[i] or squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals