#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Camarilla H4 resistance in 1w uptrend (close > EMA50) with volume spike
# - Short when price breaks below Camarilla L4 support in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or price reverts to Camarilla pivot (H3/L3)
# - Targets 15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by using 1w trend filter to align with higher timeframe momentum

name = "1d_1w_camarilla_breakout_volume_trend_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d indicators
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
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
    
    # 1d volume confirmation: > 1.8x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    
    # 1d Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5*(high-low)*1.1/2
    # L4 = close - 1.5*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d) * 1.1 / 2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike_1d[i]) or 
            np.isnan(atr_14_1d[i]) or np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to Camarilla H3 (mean reversion)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                prices['close'].iloc[i] > camarilla_h3[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to Camarilla L3 (mean reversion)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                prices['close'].iloc[i] < camarilla_l3[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla H4/L4 breakout with trend and volume filters
            if vol_spike_1d[i]:
                # Long signal: price breaks above H4 resistance in 1w uptrend
                if (prices['high'].iloc[i] >= camarilla_h4[i] and 
                    prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d[i]
                    signals[i] = 0.25
                # Short signal: price breaks below L4 support in 1w downtrend
                elif (prices['low'].iloc[i] <= camarilla_l4[i] and 
                      prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d[i]
                    signals[i] = -0.25
    
    return signals