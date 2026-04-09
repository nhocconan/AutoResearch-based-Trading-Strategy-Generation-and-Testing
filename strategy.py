#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla pivot breakouts with volume confirmation
# Camarilla pivots provide intraday support/resistance levels based on previous day's range
# Breakouts above/below key levels (H3/L3) with volume > 1.5x 20-period average indicate strong momentum
# Uses 4h trend filter (price > EMA50) to avoid counter-trend trades
# Session filter (08-20 UTC) reduces noise during low-volume periods
# Discrete position sizing 0.20 targets 15-30 trades/year to minimize fee drag
# Works in bull/bear markets: breakouts capture strong moves, filters reduce whipsaws

name = "1h_4h_1d_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (use shift(1) to avoid look-ahead)
    # Camarilla levels based on previous day's range
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    range_1d = prev_high - prev_low
    camarilla_h3 = prev_close + range_1d * 1.1 / 4
    camarilla_l3 = prev_close - range_1d * 1.1 / 4
    camarilla_h4 = prev_close + range_1d * 1.1 / 2
    camarilla_l4 = prev_close - range_1d * 1.1 / 2
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA(50) for trend filter
    close_s_4h = pd.Series(close_4h)
    ema50_4h = close_s_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume filter (20-period average)
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma20[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * volume_ma20[i]
        
        if position == 1:  # Long position
            # Exit conditions: price below L3 or trend filter fails
            if close[i] < camarilla_l3_aligned[i] or close[i] <= ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: price above H3 or trend filter fails (for shorts, we want price < EMA)
            if close[i] > camarilla_h3_aligned[i] or close[i] >= ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry conditions with volume confirmation and trend filter
            if volume_confirm:
                # Long breakout: price above H3 with uptrend (price > EMA50)
                if high[i] > camarilla_h3_aligned[i] and close[i] > ema50_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short breakdown: price below L3 with downtrend (price < EMA50)
                elif low[i] < camarilla_l3_aligned[i] and close[i] < ema50_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals