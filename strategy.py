#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + (H-L) * 1.500, R3 = C + (H-L) * 1.250, R2 = C + (H-L) * 1.166
    # S1 = C - (H-L) * 1.166, S2 = C - (H-L) * 1.250, S3 = C - (H-L) * 1.500
    # where C = (H+L+C)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    
    r3 = typical_price + hl_range * 1.250
    s3 = typical_price - hl_range * 1.250
    r4 = typical_price + hl_range * 1.500
    s4 = typical_price - hl_range * 1.500
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 6h timeframe
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: Break above R3 with volume in daily uptrend
            if close[i] > r3_aligned[i] and vol_condition and ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume in daily downtrend
            elif close[i] < s3_aligned[i] and vol_condition and ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below S3 or trend reverses
            if close[i] < s3_aligned[i] or ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above R3 or trend reverses
            if close[i] > r3_aligned[i] or ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Camarilla R3/S3 breakout with daily trend filter and volume confirmation
# - Camarilla R3/S3 levels act as key intraday support/resistance derived from prior day's range
# - Breakout above R3 with volume = bullish continuation; breakdown below S3 = bearish continuation
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (2x average) filters out weak breakouts
# - Exit when price returns to opposite S3/R3 level or trend reverses
# - Works in both bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend)
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Novelty: Camarilla levels (not commonly used in 6h) + volume spike + daily trend filter
# - Focus on BTC/ETH as primary targets; avoids over-optimization on SOL
# - Position size 0.25 balances risk/reward while minimizing transaction costs
# - Uses proper MTF handling: daily data loaded once, aligned with look-ahead prevention