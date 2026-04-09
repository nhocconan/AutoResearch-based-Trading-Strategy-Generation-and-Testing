#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels with 1w volume confirmation and chop regime filter
# - Uses weekly Camarilla pivot levels (H3/L3) for breakout signals
# - Confirms with 1w volume > 2.0x 4-period average (strong institutional participation)
# - Filters by 1w choppiness index: trade only when CHOP > 61.8 (range) OR CHOP < 38.2 (trend)
# - Exits when price touches opposite Camarilla level (H3/L3) or ATR-based stoploss (2.0x ATR)
# - Position size: 0.25 (25% of capital) for conservative risk management
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years) to minimize fee drag
# - Camarilla pivots provide mathematical support/resistance levels that work in all market regimes
# - Volume confirmation ensures breakouts have conviction
# - Chop regime filter avoids false signals in choppy markets while allowing trending/range strategies

name = "12h_1w_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate True Range for ATR and chop
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value has no previous close
    
    # 1w ATR(4) for stoploss
    atr_1w = pd.Series(tr).rolling(window=4, min_periods=4).mean().values
    
    # 1w Camarilla levels (based on previous week's OHLC)
    # H4 = Close + 1.5*(High-Low), H3 = Close + 1.0*(High-Low), etc.
    # Using previous week's values (already closed)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]  # First week: use current values
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Camarilla levels
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low)  # H3 level
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low)  # L3 level
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)  # H4 level (stoploss reference)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)  # L4 level (stoploss reference)
    
    # 1w Volume > 2.0x 4-period average (stricter for fewer trades)
    avg_volume_4 = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume_1w > (2.0 * avg_volume_4)
    
    # 1w Choppiness Index(4)
    sum_tr_4 = pd.Series(tr).rolling(window=4, min_periods=4).sum().values
    highest_4 = pd.Series(high_1w).rolling(window=4, min_periods=4).max().values
    lowest_4 = pd.Series(low_1w).rolling(window=4, min_periods=4).min().values
    chop_denom = np.where((highest_4 - lowest_4) > 0, highest_4 - lowest_4, 1e-10)
    chop = 100 * np.log10(sum_tr_4 / chop_denom) / np.log10(4)
    chop_range = chop > 61.8  # range-bound market
    chop_trend = chop < 38.2  # trending market
    
    # Align all 1w indicators to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, df_1w, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, df_1w, chop_trend.astype(float))
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_range_aligned[i]) or
            np.isnan(chop_trend_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            atr_1w_aligned[i] <= 0):
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
                atr_stop = atr_1w_aligned[i]
                signals[i] = 0.25
            elif (low[i] <= camarilla_l3_aligned[i] and   # Break below L3
                  volume_spike_aligned[i] and         # Volume confirmation
                  (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = -1
                entry_price = low[i]
                atr_stop = atr_1w_aligned[i]
                signals[i] = -0.25
    
    return signals