#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Long when price breaks above H3 level with 4h volume > 1.5x average AND daily close > daily EMA50
# - Short when price breaks below L3 level with 4h volume > 1.5x average AND daily close < daily EMA50
# - Exit when price retreats to pivot point (PP) or volume drops below average
# - Uses 4h for signal direction confirmation, 1h only for entry timing precision
# - Session filter (08-20 UTC) to avoid low-liquidity periods
# - Discrete position sizing (0.20) to minimize fee churn
# - Targets 15-35 trades/year (60-140 total over 4 years) to stay within fee drag limits

name = "1h_4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume_20_avg_4h = df_4h['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = df_4h['volume'] > (1.5 * volume_20_avg_4h)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h.astype(float))
    
    # Pre-compute 4h volume filter: < average volume for exit
    vol_normal_4h = df_4h['volume'] < volume_20_avg_4h
    vol_normal_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_normal_4h.astype(float))
    
    # Pre-compute Camarilla pivot points using previous day's OHLC
    # We need to shift by 1 to avoid look-ahead (use previous day's data)
    high_shift = prices['high'].shift(1)
    low_shift = prices['low'].shift(1)
    close_shift = prices['close'].shift(1)
    
    # Calculate pivot point and levels
    pp = (high_shift + low_shift + close_shift) / 3.0
    range_hl = high_shift - low_shift
    
    # Camarilla levels
    h3 = pp + (range_hl * 1.1 / 4)
    l3 = pp - (range_hl * 1.1 / 4)
    h4 = pp + (range_hl * 1.1 / 2)
    l4 = pp - (range_hl * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid or outside session
        if (np.isnan(pp.iloc[i]) or np.isnan(h3.iloc[i]) or np.isnan(l3.iloc[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike_4h_aligned[i]) or
            np.isnan(vol_normal_4h_aligned[i]) or not in_session.iloc[i]):
            # Hold current position if invalid data, but only if we have a position
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > H3 with 4h volume spike AND daily uptrend
            if (prices['high'].iloc[i] > h3.iloc[i] and 
                vol_spike_4h_aligned[i] > 0.5 and  # Boolean aligned as 0.0/1.0
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short breakdown: price < L3 with 4h volume spike AND daily downtrend
            elif (prices['low'].iloc[i] < l3.iloc[i] and 
                  vol_spike_4h_aligned[i] > 0.5 and
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price retreats to pivot point (PP) - mean reversion signal
            # 2. 4h volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (prices['low'].iloc[i] <= pp.iloc[i] or 
                    vol_normal_4h_aligned[i] > 0.5):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20  # Hold long
            elif position == -1:  # Short position
                if (prices['high'].iloc[i] >= pp.iloc[i] or 
                    vol_normal_4h_aligned[i] > 0.5):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20  # Hold short
    
    return signals