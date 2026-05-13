# 1d_Weekly_Camarilla_R1S1_Breakout_Trend_Filter
# Hypothesis: Enter long when price breaks above weekly Camarilla R1 level in the direction of weekly EMA34 trend, with volume confirmation. Enter short when price breaks below weekly S1 level in the direction of weekly EMA34 trend, with volume confirmation. Uses weekly levels for structure and daily timeframe for execution. Works in bull (breaks above R1 in uptrend) and bear (breaks below S1 in downtrend). Low frequency due to weekly level breaks and trend filter.

name = "1d_Weekly_Camarilla_R1S1_Breakout_Trend_Filter"
timeframe = "1d"
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

    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly OHLC for Camarilla calculation
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Camarilla levels: R1, S1 based on previous week
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = close_w + (high_w - low_w) * 1.1 / 12
    s1 = close_w - (high_w - low_w) * 1.1 / 12
    
    # Weekly trend: EMA34
    ema34_w = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema34_w_aligned = align_htf_to_ltf(prices, df_1w, ema34_w)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly R1 + weekly uptrend + volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema34_w_aligned[i] and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly S1 + weekly downtrend + volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema34_w_aligned[i] and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly EMA34 OR below weekly S1 (stop)
            if close[i] < ema34_w_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly EMA34 OR above weekly R1 (stop)
            if close[i] > ema34_w_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals