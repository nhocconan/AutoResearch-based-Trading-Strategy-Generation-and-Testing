#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation.
# Long when price breaks above R3 with volume > 1.5x average AND price > 1d EMA34.
# Short when price breaks below S3 with volume > 1.5x average AND price < 1d EMA34.
# Exit when price retouches the 1d EMA34 or Alligator alignment reverses (using Williams Alligator on 4h).
# Uses discrete position sizing (0.30) to balance return and drawdown.
# Designed for low trade frequency (~20-50/year) by requiring confluence of Camarilla breakout, volume spike, and 1d trend.
# Williams Alligator on 4h provides additional trend confirmation and exit signal.
# Effective in both bull and bear markets by capturing institutional breakout moves with trend and volume filters.

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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use the previous completed 1d bar to calculate levels for current 4h bar
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d[0] = close_1d[0]  # first bar
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    
    rng = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * rng
    s3 = prev_close_1d - 1.1 * rng
    
    # AlCamarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Williams Alligator on 4h for trend confirmation and exit
    median_price = (high + low) / 2
    
    # Jaw: EMA(13) of median, smoothed 8 periods
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA(8) of median, smoothed 5 periods
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA(5) of median, smoothed 3 periods
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Alligator alignment: bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw
    bullish_align = (lips > teeth) & (teeth > jaw)
    bearish_align = (lips < teeth) & (teeth < jaw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3, volume spike, price > 1d EMA34, bullish Alligator alignment
            if close[i] > r3_aligned[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i] and bullish_align[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below S3, volume spike, price < 1d EMA34, bearish Alligator alignment
            elif close[i] < s3_aligned[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i] and bearish_align[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retouches 1d EMA34 OR Alligator alignment turns bearish
            if close[i] <= ema34_1d_aligned[i] or bearish_align[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price retouches 1d EMA34 OR Alligator alignment turns bullish
            if close[i] >= ema34_1d_aligned[i] or bullish_align[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals