#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level AND close > 1d EMA34 AND volume > 1.5 * avg_volume(20).
# Short when price breaks below Camarilla S3 level AND close < 1d EMA34 AND volume > 1.5 * avg_volume(20).
# Exit when price re-enters the Camarilla H3/L3 range OR volume drops below avg_volume(20).
# Uses discrete position sizing (0.30) to balance return and drawdown.
# Designed for moderate trade frequency (~25-50/year) by requiring confluence of breakout, trend, and volume.
# Camarilla levels provide precise intraday support/resistance derived from prior day's range.
# Effective in both bull and bear markets by capturing strong directional moves with trend and volume filters.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (HLC of previous day)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(len(df_1d)):
        # Camarilla levels for day i based on day i-1's OHLC
        if i > 0:
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            range_ = prev_high - prev_low
            camarilla_r3[i] = prev_close + range_ * 1.1 / 4
            camarilla_s3[i] = prev_close - range_ * 1.1 / 4
            camarilla_h3[i] = prev_close + range_ * 1.1 / 6
            camarilla_l3[i] = prev_close - range_ * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or \
           np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: break above R3, price > 1d EMA34, volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: break below S3, price < 1d EMA34, volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price re-enters H3-L3 range OR volume drops below average
            if close[i] < camarilla_h3_aligned[i] and close[i] > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif volume[i] < vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price re-enters H3-L3 range OR volume drops below average
            if close[i] < camarilla_h3_aligned[i] and close[i] > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif volume[i] < vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals