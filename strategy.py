#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 breakouts capture institutional order flow with tighter levels than R3/S3,
1d EMA34 filters primary trend, volume spike confirms participation, and chop filter avoids extreme regimes.
Works in bull/bear via trend filter. Target: 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla H3, L3 levels (tighter than R3/S3)
    # H3 = Close + ((High-Low) * 1.1/6)
    # L3 = Close - ((High-Low) * 1.1/6)
    camarilla_h3 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    camarilla_l3 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index filter (avoid ranging and extreme trending markets)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(14))) / log10(14)
    # Simplified: high-low range relative to ATR
    true_range = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    price_range_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values - pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    price_range_14 = np.where(price_range_14 == 0, 1e-10, price_range_14)
    chop = 100 * np.log10(atr_14 * 14 / price_range_14) / np.log10(14)
    chop_filter = (chop > 30) & (chop < 70)  # Trade in moderate choppy regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34, 20, 14) + 1  # +1 for previous bar reference
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        
        # Camarilla breakout conditions
        breakout_long = curr_high > camarilla_h3_aligned[i]  # Break above H3
        breakout_short = curr_low < camarilla_l3_aligned[i]  # Break below L3
        
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + trend alignment + volume + chop filter
            long_entry = breakout_long and uptrend and vol_spike and chop_ok
            short_entry = breakout_short and downtrend and vol_spike and chop_ok
            
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
            # Exit: price retouches L3 level OR trend reverses
            if curr_close < camarilla_l3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price retouches H3 level OR trend reverses
            if curr_close > camarilla_h3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0