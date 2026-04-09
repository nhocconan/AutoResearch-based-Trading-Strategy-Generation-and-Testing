#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and chop regime filter
# - Uses Camarilla levels from 4h for entry (long above H3, short below L3)
# - Confirms with 4h volume > 1.8x 20-period average (strong participation)
# - Filters by 4h choppiness index: only trade when CHOP > 61.8 (range) or CHOP < 38.2 (trend)
# - Exits when price touches opposite Camarilla level (H3/L3) or ATR-based stoploss (2x ATR)
# - Position size: 0.20 (20% of capital) to minimize fee drag
# - Session filter: 08-20 UTC to reduce noise trades
# - Target: 60-150 total trades over 4 years = 15-37/year for 1h
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Uses 4h for signal direction, 1h only for entry timing precision

name = "1h_4h_camarilla_volume_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h True Range for ATR and chop
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 4h ATR(14) for stoploss
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h Camarilla levels (based on previous bar's range)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    range_4h = prev_high - prev_low
    camarilla_h3 = prev_close + (1.1 * range_4h * 1.1 / 4)
    camarilla_l3 = prev_close - (1.1 * range_4h * 1.1 / 4)
    camarilla_h4 = prev_close + (1.1 * range_4h * 1.1 / 2)
    camarilla_l4 = prev_close - (1.1 * range_4h * 1.1 / 2)
    
    # 4h Volume > 1.8x 20-period average
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (1.8 * avg_volume_20)
    
    # 4h Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop_range = chop > 61.8  # range-bound market
    chop_trend = chop < 38.2  # trending market
    
    # Align all 4h indicators to 1h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, df_4h, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, df_4h, chop_trend.astype(float))
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for 08-20 UTC filter
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_range_aligned[i]) or
            np.isnan(chop_trend_aligned[i]) or np.isnan(atr_4h_aligned[i]) or
            atr_4h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Camarilla touch (L3) or ATR stoploss
            if low[i] <= camarilla_l3_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Camarilla touch (H3) or ATR stoploss
            if high[i] >= camarilla_h3_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and regime filter
            if (high[i] >= camarilla_h3_aligned[i] and  # Break above H3
                volume_spike_aligned[i] and         # Volume confirmation
                (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = 1
                entry_price = high[i]
                atr_stop = atr_4h_aligned[i]
                signals[i] = 0.20
            elif (low[i] <= camarilla_l3_aligned[i] and   # Break below L3
                  volume_spike_aligned[i] and         # Volume confirmation
                  (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = -1
                entry_price = low[i]
                atr_stop = atr_4h_aligned[i]
                signals[i] = -0.20
    
    return signals