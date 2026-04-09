#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
# - Uses Camarilla levels from 12h for entry (long above H3, short below L3)
# - Confirms with 12h volume > 1.8x 20-period average (strong institutional participation)
# - Filters by 12h choppiness index: only trade when CHOP > 61.8 (range) or CHOP < 38.2 (trend)
# - Exits when price touches opposite Camarilla level (H3/L3) or ATR-based stoploss (2x ATR)
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# - Camarilla levels provide precise intraday support/resistance that work across regimes

name = "4h_12h_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h True Range for ATR and chop
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 12h ATR(14) for stoploss
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Camarilla levels (based on previous period's range)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #          L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # We use H3 and L3 as primary breakout levels
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = high_12h[0]
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    range_12h = prev_high - prev_low
    camarilla_h3 = prev_close + (1.1 * range_12h * 1.1 / 4)
    camarilla_l3 = prev_close - (1.1 * range_12h * 1.1 / 4)
    camarilla_h4 = prev_close + (1.1 * range_12h * 1.1 / 2)
    camarilla_l4 = prev_close - (1.1 * range_12h * 1.1 / 2)
    
    # 12h Volume > 1.8x 20-period average
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.8 * avg_volume_20)
    
    # 12h Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop_range = chop > 61.8  # range-bound market
    chop_trend = chop < 38.2  # trending market
    
    # Align all 12h indicators to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, df_12h, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, df_12h, chop_trend.astype(float))
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_range_aligned[i]) or
            np.isnan(chop_trend_aligned[i]) or np.isnan(atr_12h_aligned[i]) or
            atr_12h_aligned[i] <= 0):
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
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Camarilla touch (H3) or ATR stoploss
            if high[i] >= camarilla_h3_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and regime filter
            if (high[i] >= camarilla_h3_aligned[i] and  # Break above H3
                volume_spike_aligned[i] and         # Volume confirmation
                (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = 1
                entry_price = high[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = 0.25
            elif (low[i] <= camarilla_l3_aligned[i] and   # Break below L3
                  volume_spike_aligned[i] and         # Volume confirmation
                  (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = -1
                entry_price = low[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = -0.25
    
    return signals