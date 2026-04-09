#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d HTF regime filter
# - Uses 12h HTF for trend direction via EMA(20) > EMA(50) (bullish) or < (bearish)
# - Uses 1d HTF for Camarilla pivot levels (R3, R4, S3, S4)
# - Long when: 12h EMA20 > EMA50 AND price breaks above 1d R4 with volume > 1.5x 20-period average
# - Short when: 12h EMA20 < EMA50 AND price breaks below 1d S4 with volume > 1.5x 20-period average
# - Exit when price returns to 1d R3 (for longs) or S3 (for shorts) OR opposite pivot break occurs
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Combines HTF trend alignment with intraday pivot structure for high-probability breaks
# - Volume confirmation reduces false breakouts
# - Works in bull markets (long bias) and bear markets (short bias) via 12h trend filter

name = "6h_12h_1d_camarilla_breakout_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA20 and EMA50 for trend regime
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot point = (high + low + close) / 3
    # Range = high - low
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    r4 = close_1d + rng * 1.1 / 2.0
    r3 = close_1d + rng * 1.1 / 4.0
    s3 = close_1d - rng * 1.1 / 4.0
    s4 = close_1d - rng * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit conditions: price returns to R3 OR short breakout occurs
            if close[i] <= r3_aligned[i] or (volume_confirmed and close[i] < s4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price returns to S3 OR long breakout occurs
            if close[i] >= s3_aligned[i] or (volume_confirmed and close[i] > r4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: 12h trend alignment + 1d Camarilla breakout + volume
            if volume_confirmed:
                bullish_regime = ema_20_12h_aligned[i] > ema_50_12h_aligned[i]
                bearish_regime = ema_20_12h_aligned[i] < ema_50_12h_aligned[i]
                
                # Long entry: bullish 12h regime AND price breaks above 1d R4
                if bullish_regime and close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: bearish 12h regime AND price breaks below 1d S4
                elif bearish_regime and close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals