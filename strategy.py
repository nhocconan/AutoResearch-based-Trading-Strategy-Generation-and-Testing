#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + Volume Spike + 1d EMA34 Trend Filter + Chop Filter
Hypothesis: Camarilla pivot levels act as intraday support/resistance. Breakouts above H3 or below L3 with volume confirmation indicate institutional participation. 1d EMA34 filter ensures trades align with daily trend, reducing false breakouts in choppy markets. Chop filter (BW percentile < 50) avoids ranging markets. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-40 trades/year on 4h.
Works in bull markets via breakouts with trend and in bear markets via trend filter (avoids counter-trend entries).
"""

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
    
    # Get 1d data for pivot calculation and EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Bollinger Bands for chop filter (20, 2)
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma
    # Chop filter: BB width percentile < 50 (avoid extreme volatility regimes)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=100).rank(pct=True).values
    chop_filter = bb_width_percentile < 0.5  # True when in lower 50% of width (less volatile/choppy)
    
    # Calculate Camarilla pivots from previous 1d OHLC
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    rang = prev_high - prev_low
    H3 = prev_close + 1.0 * rang
    L3 = prev_close - 1.0 * rang
    
    # Align Camarilla levels to 4h (use previous day's levels for current day's trading)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA, EMA, and BB calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        is_choppy = chop_filter[i]  # True when BB width is in lower 50% (less volatile)
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > 1d EMA34 (uptrend) AND chop filter
            long_entry = (curr_close > H3_level) and vol_spike and (curr_close > ema_trend) and is_choppy
            # Short: price breaks below L3 AND volume spike AND price < 1d EMA34 (downtrend) AND chop filter
            short_entry = (curr_close < L3_level) and vol_spike and (curr_close < ema_trend) and is_choppy
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L3 (reversal) OR price < 1d EMA34 (trend change)
            if (curr_close < L3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 (reversal) OR price > 1d EMA34 (trend change)
            if (curr_close > H3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_VolumeSpike_1dEMA34_Trend_ChopFilter"
timeframe = "4h"
leverage = 1.0