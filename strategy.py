#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Camarilla H4 resistance in 12h uptrend with volume spike
# - Short when price breaks below Camarilla L4 support in 12h downtrend with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Exit when price reverts to Camarilla pivot (mean reversion) or ATR-based stoploss
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag

name = "6h_12h_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h ATR(14) for stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_12h = np.zeros_like(tr)
    atr_14_12h[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_12h[i] = (atr_14_12h[i-1] * (14-1) + tr[i]) / 14
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # 12h volume confirmation: > 1.5x 20-period average
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * avg_volume_20_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # 12h Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.1*(high-low)*1.1/2
    # L4 = close - 1.1*(high-low)*1.1/2
    camarilla_h4 = close_12h + 1.1 * (high_12h - low_12h) * 1.1 / 2
    camarilla_l4 = close_12h - 1.1 * (high_12h - low_12h) * 1.1 / 2
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    camarilla_pivot = (high_12h + low_12h + close_12h) / 3
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(atr_14_12h_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to Camarilla pivot (mean reversion)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] < camarilla_pivot_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to Camarilla pivot (mean reversion)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] > camarilla_pivot_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla H4/L4 breakout with trend and volume filters
            if vol_spike_12h_aligned[i]:
                # Long signal: price breaks above H4 resistance in 12h uptrend
                if (prices['high'].iloc[i] >= camarilla_h4_aligned[i] and 
                    prices['close'].iloc[i] > ema_50_12h_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_12h_aligned[i]
                    signals[i] = 0.25
                # Short signal: price breaks below L4 support in 12h downtrend
                elif (prices['low'].iloc[i] <= camarilla_l4_aligned[i] and 
                      prices['close'].iloc[i] < ema_50_12h_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_12h_aligned[i]
                    signals[i] = -0.25
    
    return signals