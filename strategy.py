#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter
# Williams Alligator (jaw/teeth/lips) identifies trend absence/presence via SMAs
# Elder Ray (Bull/Bear power = EMA13 - high/low) measures trend strength
# 1d EMA50 filter ensures trades align with higher timeframe trend
# Works in bull markets (Alligator awake + Bull power > 0) and bear markets (Alligator awake + Bear power < 0)
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) on 1d close
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Williams Alligator: SMAs of median price (typical price = (H+L+C)/3)
    typical_price = (high + low + close) / 3.0
    
    # Jaw: SMA(13, 8) - 13-period SMA shifted 8 bars forward
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA(8, 5) - 8-period SMA shifted 5 bars forward
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA(5, 3) - 5-period SMA shifted 3 bars forward
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators (max shift is 8 for jaw)
    start_idx = 13  # Need 13 for EMA13 and SMAs
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator conditions: 
        # - Mouth open (teeth above/below lips) indicates trend
        # - All lines aligned: jaw < teeth < lips for uptrend, jaw > teeth > lips for downtrend
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        alligator_long = jaw_val < teeth_val < lips_val  # Mouth open up
        alligator_short = jaw_val > teeth_val > lips_val  # Mouth open down
        
        # Elder Ray conditions:
        # - Long: Bull power > 0 and increasing (current > previous)
        # - Short: Bear power < 0 and decreasing (current < previous)
        bull_long = bull_power[i] > 0 and (i == start_idx or bull_power[i] > bull_power[i-1])
        bear_short = bear_power[i] < 0 and (i == start_idx or bear_power[i] < bear_power[i-1])
        
        # 1d trend filter:
        # - Long: price above 1d EMA50
        # - Short: price below 1d EMA50
        trend_long = close[i] > ema_1d_50_aligned[i]
        trend_short = close[i] < ema_1d_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator mouth open up + Bull power positive + Uptrend on 1d
            if alligator_long and bull_long and trend_long:
                signals[i] = 0.25
                position = 1
            # Short: Alligator mouth open down + Bear power negative + Downtrend on 1d
            elif alligator_short and bear_short and trend_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator mouth closes OR Bear power turns negative OR price below 1d EMA50
            if not alligator_long or not bull_long or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator mouth closes OR Bull power turns positive OR price above 1d EMA50
            if not alligator_short or not bear_short or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals