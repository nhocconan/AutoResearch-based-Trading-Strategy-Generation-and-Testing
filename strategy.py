#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm
Hypothesis: Fade at Camarilla R3/S3 levels with 1d trend filter and volume confirmation works in both bull and bear markets by capturing mean reversion at extreme intraday levels while avoiding choppy markets. Designed for low trade frequency (~12-37/year) on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA on 1d ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla levels from prior 1d session (HLC of previous 1d bar) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels (fade signals)
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume spike filter (20-period on 6h, using 1d data as proxy for simplicity) ===
    # Since we don't have intraday volume in 1d data, we'll use price range as volatility proxy
    range_1d = high_1d - low_1d
    range_ma_1d = pd.Series(range_1d).rolling(window=20, min_periods=20).mean().values
    range_ratio_1d = range_1d / range_ma_1d
    range_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, range_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(range_ratio_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_proxy = range_ratio_1d_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long fade: price drops to S3 + volume proxy > 1.2 + price below 1d EMA (bearish bias)
            if price_low <= s3 and vol_proxy > 1.2 and price_close < trend_1d:
                signals[i] = 0.25
                position = 1
            # Short fade: price rises to R3 + volume proxy > 1.2 + price above 1d EMA (bullish bias)
            elif price_high >= r3 and vol_proxy > 1.2 and price_close > trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: mean reversion complete or trend reversal
            if position == 1:  # Long position
                # Exit when price reaches midpoint (mean reversion target) or trend turns bullish
                midpoint = (r3 + s3) / 2
                if price_close >= midpoint or price_close > trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit when price reaches midpoint (mean reversion target) or trend turns bearish
                midpoint = (r3 + s3) / 2
                if price_close <= midpoint or price_close < trend_1d:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0