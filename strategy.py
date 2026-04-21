#!/usr/bin/env python3
"""
1d_Range_Mean_Reversion_With_Weekly_Trend_Filter
Hypothesis: On 1d timeframe, trade mean-reversion at Bollinger Bands (20,2) when price touches bands in ranging markets (BBW percentile < 50), filtered by weekly trend (price above/below weekly 50 EMA). In ranging markets, price tends to revert to mean; in trending markets, follow the trend. Weekly EMA filter prevents counter-trend trades during strong trends. Designed for 1d to achieve 7-25 trades/year with low frequency and high edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly trend filter: 50-period EMA ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Bollinger Bands (20,2) on 1d ===
    close = prices['close'].values
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Width for regime detection
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 50, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(bb_middle[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i]) or
            np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        weekly_trend = ema_50_1w_aligned[i]
        bw_percentile = bb_width_percentile[i]
        
        # Regime: ranging if BB width percentile < 50 (narrow bands)
        is_ranging = bw_percentile < 50
        
        if position == 0:
            if is_ranging:
                # In ranging markets: mean reversion at Bollinger Bands
                if price_close <= bb_lower[i]:
                    signals[i] = 0.25
                    position = 1
                elif price_close >= bb_upper[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                # In trending markets: follow weekly trend
                if price_close > weekly_trend:
                    signals[i] = 0.25
                    position = 1
                elif price_close < weekly_trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price returns to Bollinger middle (mean)
            if position == 1 and price_close >= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close <= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Range_Mean_Reversion_With_Weekly_Trend_Filter"
timeframe = "1d"
leverage = 1.0