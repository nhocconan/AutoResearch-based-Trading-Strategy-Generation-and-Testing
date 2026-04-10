#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot with 1d trend filter and volume spike
# - Long when price touches Camarilla L3 support in 1d uptrend (close > EMA50) with volume > 1.8x 20-period average
# - Short when price touches Camarilla H3 resistance in 1d downtrend (close < EMA50) with volume spike
# - Exit on ATR(14) stoploss (2.0x) or reversion to Camarilla H4/L4 levels
# - Discrete position sizing (0.25) to minimize fee churn
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
# - Works in bull/bear: trend filter captures direction, Camarilla provides mean-reversion entries in ranging markets

name = "12h_1d_camarilla_pivot_volume_trend_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter (faster than 200 for 12h chart)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d volume confirmation: > 1.8x 20-period average (stricter to reduce trades)
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Camarilla pivot levels (based on previous day)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 2
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to Camarilla H4 (strong resistance)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] > camarilla_h4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to Camarilla L4 (strong support)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] < camarilla_l4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla L3/H3 touch with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: price touches L3 support in 1d uptrend
                if (prices['low'].iloc[i] <= camarilla_l3_aligned[i] and 
                    prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = 0.25
                # Short signal: price touches H3 resistance in 1d downtrend
                elif (prices['high'].iloc[i] >= camarilla_h3_aligned[i] and 
                      prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = -0.25
    
    return signals