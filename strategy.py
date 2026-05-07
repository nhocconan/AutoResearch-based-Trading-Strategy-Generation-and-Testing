# 4H_CAMARILLA_R1_S1_12HTREND_VOLUME_SPIKE
# Hypothesis: Camarilla pivot breakout with 12h trend filter and volume spike
# Uses daily Camarilla R1/S1 levels for entry, 12h EMA50 for trend filter,
# and volume spike (2x 20-period average) for confirmation. Designed to work
# in both bull and bear markets by combining strong support/resistance levels
# with trend alignment and volume confirmation. Targets 15-30 trades/year to
# avoid fee drag while maintaining edge in BTC/ETH.

name = "4H_Camarilla_R1_S1_12HTrend_Volume_Spike"
timeframe = "4h"
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
    
    # Load daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align daily Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R1 in 12h uptrend with volume spike
            if close[i] > r1_aligned[i] and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 in 12h downtrend with volume spike
            elif close[i] < s1_aligned[i] and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below S1 or trend weakens
            if close[i] < s1_aligned[i] or ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above R1 or trend weakens
            if close[i] > r1_aligned[i] or ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla pivot breakout with 12h trend filter and volume spike
# - Uses daily Camarilla R1/S1 levels as key support/resistance levels
# - Enters long when price breaks above R1 with volume spike in 12h uptrend
# - Enters short when price breaks below S1 with volume spike in 12h downtrend
# - Exits when price returns to opposite level or trend weakens
# - Volume spike (2x average) confirms breakout validity
# - Position size 0.25 balances return and risk
# - Designed for 15-30 trades/year to avoid fee drag while maintaining edge
# - Works in both bull (R1 breakouts in uptrend) and bear (S1 breakdowns in downtrend)
# - Combines proven elements: Camarilla pivots, trend filtering, volume confirmation
# - Avoids overtrading through multiple confirmation requirements
# - Tested on BTC/ETH/SOL with focus on major pairs for robustness