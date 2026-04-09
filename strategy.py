#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# - Uses weekly Camarilla pivot levels (H3/L3) to determine bias: long if price > weekly H3, short if price < weekly L3
# - Entry on 6h Donchian(20) breakout in direction of weekly bias
# - Confirmed by 6h volume > 1.8x 20-period average (institutional participation)
# - Exits when price touches opposite Donchian band or ATR-based stop (2.5x ATR)
# - Position size: 0.25 (25% of capital) to manage drawdown in volatile markets
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to minimize fee drag
# - Weekly Camarilla provides structural bias that adapts to volatility and works in both bull/bear markets
# - Donchian breakouts capture momentum; volume confirmation filters false breakouts

name = "6h_1w_donchian_volume_camarilla_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute weekly indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly True Range for ATR
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr_1w[0]
    
    # Weekly ATR(14) for Camarilla calculation
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Weekly Camarilla pivot levels (H3/L3 for bias, H4/L4 for extreme levels)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    camarilla_h3 = typical_price_1w + (range_1w * 1.1 / 4)
    camarilla_l3 = typical_price_1w - (range_1w * 1.1 / 4)
    camarilla_h4 = typical_price_1w + (range_1w * 1.1 / 2)
    camarilla_l4 = typical_price_1w - (range_1w * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Pre-compute 6h indicators
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # 6h True Range for ATR
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr_6h[0]
    
    # 6h ATR(14) for stoploss
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # 6h Donchian(20) channels
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # 6h Volume > 1.8x 20-period average
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume_6h > (1.8 * avg_volume_20)
    
    # Align 6h indicators to primary timeframe
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    volume_spike_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_spike_6h.astype(float))
    
    # Primary timeframe price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike_6h_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            atr_6h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Determine weekly bias: long if price > weekly H3, short if price < weekly L3
        weekly_long_bias = close[i] > camarilla_h3_aligned[i]
        weekly_short_bias = close[i] < camarilla_l3_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: opposite Donchian touch or ATR stoploss
            if low[i] <= donchian_low_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Donchian touch or ATR stoploss
            if high[i] >= donchian_high_aligned[i]:  # Touch opposite band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.5 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout in direction of weekly bias with volume confirmation
            if (high[i] >= donchian_high_aligned[i] and  # Break above upper band
                weekly_long_bias and                   # Weekly bias long
                volume_spike_6h_aligned[i]):           # Volume confirmation
                position = 1
                entry_price = high[i]
                atr_stop = atr_6h_aligned[i]
                signals[i] = 0.25
            elif (low[i] <= donchian_low_aligned[i] and   # Break below lower band
                  weekly_short_bias and                 # Weekly bias short
                  volume_spike_6h_aligned[i]):          # Volume confirmation
                position = -1
                entry_price = low[i]
                atr_stop = atr_6h_aligned[i]
                signals[i] = -0.25
    
    return signals