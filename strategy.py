# 1d_1w_Camarilla_Pivot_VolumeSpike_Trend_v1
# Hypothesis: Daily breakouts from weekly Camarilla H3/L3 levels with volume spike and weekly trend filter.
# Weekly trend uses 50 EMA on weekly timeframe to filter trades in direction of higher timeframe trend.
# Volume spike requires 2x 20-period average to confirm institutional interest.
# Targets 15-25 trades/year to minimize fee drag while capturing significant breakouts.
# Works in bull markets via trend-following breaks and in bear via mean-reversion at extreme levels.

name = "1d_1w_Camarilla_Pivot_VolumeSpike_Trend_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous weekly bar
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Avoid look-ahead: only use previous week's data
    range_ = prev_high - prev_low
    H3 = prev_close + 1.1 * range_ / 4
    L3 = prev_close - 1.1 * range_ / 4
    
    # Align Camarilla levels to daily timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(H3_aligned[i]) or
            np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above H3 + volume spike + price above weekly EMA50
            if (close[i] > H3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below L3 + volume spike + price below weekly EMA50
            elif (close[i] < L3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters weekly range (between L3 and H3) OR closes below weekly EMA50
            if (close[i] > L3_aligned[i] and close[i] < H3_aligned[i]) or \
               close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters weekly range (between L3 and H3) OR closes above weekly EMA50
            if (close[i] > L3_aligned[i] and close[i] < H3_aligned[i]) or \
               close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals