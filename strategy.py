#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop regime filter
# - Uses 1d Camarilla pivot levels (H3/L3) for breakout entries
# - Requires volume > 1.5x 20-period average for confirmation
# - Uses 4h choppiness index (CHOP > 61.8 = ranging, CHOP < 38.2 = trending) as regime filter
# - Only takes breakout trades in trending regime (CHOP < 38.2)
# - ATR-based stoploss and trailing exit via signal=0 when price reverses
# - Designed for 4h timeframe to balance trade frequency and edge capture
# - Target: 20-40 trades/year to avoid fee drag while capturing meaningful moves

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute chop regime filter on 4h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range and ATR for chop calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(atr14) / (n * log10(highest_high - lowest_low)))
    # Simplified: CHOP = 100 * log10(sum(atr14 over period) / (log10(HHLL) * period))
    sum_atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    hhll = highest_high - lowest_low
    # Avoid division by zero and log of zero/negative
    valid = (hhll > 0) & (sum_atr14 > 0)
    chop = np.full(n, 50.0)  # default to neutral
    chop[valid] = 100 * np.log10(sum_atr14[valid] / (np.log10(hhll[valid]) * 14))
    # Regime: CHOP < 38.2 = trending (favor breakouts), CHOP > 61.8 = ranging (avoid)
    trending_regime = chop < 38.2
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma20)
    
    # 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3, H4, L3, L4
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(vol_ma20[i]) or vol_ma20[i] <= 0 or
            np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below L3 or regime changes to ranging
            if close[i] < l3_aligned[i] or not trending_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above H3 or regime changes to ranging
            if close[i] > h3_aligned[i] or not trending_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries in trending regime with volume confirmation
            if (close[i] > h3_aligned[i] and 
                trending_regime[i] and 
                volume_confirm[i]):
                position = 1
                signals[i] = 0.25
            elif (close[i] < l3_aligned[i] and 
                  trending_regime[i] and 
                  volume_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals