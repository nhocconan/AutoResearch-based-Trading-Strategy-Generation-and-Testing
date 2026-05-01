#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. In strong trends, power persists.
# Uses 12h EMA50 for trend direction (long only when price > EMA50, short when < EMA50).
# Volume spike confirms institutional participation. Works in bull (trend continuation) and bear (mean reversion at extremes).

name = "6h_ElderRay_Power_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 (using prior close to avoid look-ahead)
    close_12h = df_12h['close'].values
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA)
    # Calculate EMA13 on 6h close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth the power values with EMA3 to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short