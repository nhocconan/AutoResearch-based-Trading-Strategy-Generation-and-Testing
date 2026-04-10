#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot mean reversion with 4h trend filter and volume confirmation
# - Long when price touches Camarilla L3 support in 4h uptrend (close > EMA50) with volume spike
# - Short when price touches Camarilla H3 resistance in 4h downtrend (close < EMA50) with volume spike
# - Exit when price reverts to Camarilla pivot (mean reversion) or ATR-based stoploss
# - Uses 1h timeframe for precise entry timing, 4h for signal direction to reduce overtrading
# - Targets 15-37 trades/year (60-150 total over 4 years) to avoid fee drag
# - Session filter: 08-20 UTC to avoid low-volume noisy periods

name = "1h_4h_camarilla_pivot_meanrev_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours once (08-20 UTC)
    hours = prices.index.hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 100:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_4h = np.zeros_like(tr)
    atr_14_4h[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_4h[i] = (atr_14_4h[i-1] * (14-1) + tr[i]) / 14
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # 4h volume confirmation: > 1.5x 20-period average
    avg_volume_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume_4h > (1.5 * avg_volume_20_4h)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h)
    
    # 4h Camarilla pivot levels (based on previous day)
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h) * 1.1 / 4
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h) * 1.1 / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_pivot = (high_4h + low_4h + close_4h) / 3
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_spike_4h_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to Camarilla pivot (mean reversion)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] > camarilla_pivot_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to Camarilla pivot (mean reversion)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] < camarilla_pivot_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla L3/H3 touch with trend and volume filters
            if vol_spike_4h_aligned[i]:
                # Long signal: price touches L3 support in 4h uptrend
                if (prices['low'].iloc[i] <= camarilla_l3_aligned[i] and 
                    prices['close'].iloc[i] > ema_50_4h_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h_aligned[i]
                    signals[i] = 0.20
                # Short signal: price touches H3 resistance in 4h downtrend
                elif (prices['high'].iloc[i] >= camarilla_h3_aligned[i] and 
                      prices['close'].iloc[i] < ema_50_4h_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_4h_aligned[i]
                    signals[i] = -0.20
    
    return signals