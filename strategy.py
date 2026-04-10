#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 12h volume > 1.8x 20-period average AND chop > 61.8 (range regime)
# - Short when price breaks below Camarilla L3 level AND 12h volume > 1.8x 20-period average AND chop > 61.8 (range regime)
# - Exit when price returns to Camarilla H4/L4 levels (mean reversion in range)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Camarilla pivots work well in ranging markets (2025-2026 bear/range)
# - Volume confirmation reduces false breakouts
# - Chop filter ensures we trade in ranging regimes where mean reversion works

name = "4h_12h_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4, etc.
    # We need daily high/low/close from 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    camarilla_h5 = np.zeros_like(close_1d)  # Stop loss level
    camarilla_l5 = np.zeros_like(close_1d)  # Stop loss level
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_h4[i] = camarilla_h3[i] = camarilla_l3[i] = camarilla_l4[i] = camarilla_h5[i] = camarilla_l5[i] = np.nan
        else:
            rng = high_1d[i-1] - low_1d[i-1]
            camarilla_h4[i] = close_1d[i-1] + rng * 1.1 * 1.1 / 2
            camarilla_h3[i] = close_1d[i-1] + rng * 1.1 * 1.1 / 4
            camarilla_l3[i] = close_1d[i-1] - rng * 1.1 * 1.1 / 4
            camarilla_l4[i] = close_1d[i-1] - rng * 1.1 * 1.1 / 2
            camarilla_h5[i] = close_1d[i-1] + rng * 1.1 * 1.1  # Extended stop loss
            camarilla_l5[i] = close_1d[i-1] - rng * 1.1 * 1.1  # Extended stop loss
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h5_4h = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_4h = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = vol_12h > (1.8 * vol_ma_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    # Pre-compute 4h Choppiness Index (CHOP) for regime filter
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # First bar
    tr = true_range(high, low, prev_close)
    
    # ATR(14) for CHOP denominator
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[1:14])  # First ATR
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Sum of TRUE RANGE over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (atr_14 * 14)) / log10(14)
    chop = np.zeros_like(close)
    for i in range(14, len(close)):
        if atr_14[i] > 0 and tr_sum_14[i] > 0:
            chop[i] = 100 * np.log10(tr_sum_14[i] / (atr_14[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # Neutral value
    
    # Chop regime: > 61.8 = ranging (good for mean reversion)
    chop_regime = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_4h[i]) or np.isnan(camarilla_l3_4h[i]) or 
            np.isnan(camarilla_h4_4h[i]) or np.isnan(camarilla_l4_4h[i]) or
            np.isnan(volume_spike_12h_aligned[i]) or np.isnan(chop_regime[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND chop regime (ranging)
            if (close[i] > camarilla_h3_4h[i] and 
                volume_spike_12h_aligned[i] and 
                chop_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND chop regime (ranging)
            elif (close[i] < camarilla_l3_4h[i] and 
                  volume_spike_12h_aligned[i] and 
                  chop_regime[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Exit conditions: price returns to H4/L4 levels (mean reversion target)
            exit_long = (position == 1 and close[i] < camarilla_h4_4h[i])
            exit_short = (position == -1 and close[i] > camarilla_l4_4h[i])
            
            # Stoploss: extended H5/L5 levels
            stop_long = (position == 1 and close[i] > camarilla_h5_4h[i])
            stop_short = (position == -1 and close[i] < camarilla_l5_4h[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals