#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (H3/L3) breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla levels provide precise intraday support/resistance derived from prior day's range
# H3 breakout = bullish continuation, L3 breakdown = bearish continuation
# 1w EMA50 ensures we only trade with the higher timeframe trend (avoids counter-trend whipsaws)
# Volume spike (>1.8x 24-period EMA) confirms institutional participation
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Works in bull/bear markets by following the 1w trend direction via EMA50

name = "12h_Camarilla_H3L3_Breakout_1wEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # 1d HTF data for Camarilla pivot calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w HTF data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h EMA24 for volume filter
    close_s = pd.Series(close)
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    
    # Precompute Camarilla levels for each 12h bar using prior 1d OHLC
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # where C = prior day close, H = prior day high, L = prior day low
    prior_close = df_1d['close'].shift(1).values  # prior day close
    prior_high = df_1d['high'].shift(1).values    # prior day high
    prior_low = df_1d['low'].shift(1).values      # prior day low
    
    # Align prior day OHLC to 12h timeframe
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    
    # Calculate Camarilla H3 and L3 levels
    rango = prior_high_aligned - prior_low_aligned
    camarilla_h3 = prior_close_aligned + (rango * 1.1 / 4)
    camarilla_l3 = prior_close_aligned - (rango * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 1w EMA50, prior day OHLC, volume EMA)
    start_idx = max(50, 1, 24)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(prior_close_aligned[i]) or 
            np.isnan(prior_high_aligned[i]) or np.isnan(prior_low_aligned[i]) or
            np.isnan(vol_ema_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend:
                # Long: price breaks above Camarilla H3 with volume spike
                if close[i] > camarilla_h3[i] and volume[i] > (1.8 * vol_ema_24[i]):
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend:
                # Short: price breaks below Camarilla L3 with volume spike
                if close[i] < camarilla_l3[i] and volume[i] > (1.8 * vol_ema_24[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid sideways markets
        
        elif position == 1:  # Long position
            # Exit: price returns below Camarilla H3 (failure of breakout)
            if close[i] < camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns above Camarilla L3 (failure of breakdown)
            if close[i] > camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals