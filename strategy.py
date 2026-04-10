#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + chop regime filter
# - Uses 1d Camarilla levels (H3, L3, H4, L4) as institutional support/resistance
# - Long when price crosses above L3 with volume confirmation (>1.5x 20-bar avg)
# - Short when price crosses below H3 with volume confirmation
# - Choppiness index regime filter: only trade when CHOP(14) < 61.8 (trending market)
# - Designed for 4h timeframe: targets 20-50 trades/year to avoid fee drag
# - Works in bull/bear markets: Camarilla levels adapt to volatility, chop filter avoids ranging markets
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(20)

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    H3 = pivot + (range_hl * 1.1 / 4.0)
    L3 = pivot - (range_hl * 1.1 / 4.0)
    H4 = pivot + (range_hl * 1.1 / 2.0)
    L4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Align HTF Camarilla levels to LTF (4h)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Pre-compute 4h ATR(20) for stoploss
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * avg_volume_20)
    
    # Pre-compute 4h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(TR over n) / (max(HH,n) - min(LL,n))) / log10(n)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high - lowest_low
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(atr_sum / chop_denom) / np.log10(14)
    chop = np.where(chop_denom <= 0, 50.0, chop)  # default to neutral when invalid
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(atr_20[i]) or np.isnan(vol_spike[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price crosses below L3 (support break)
            if prices['close'].iloc[i] < entry_price - 2.0 * atr_20[i] or prices['close'].iloc[i] < L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price crosses above H3 (resistance break)
            if prices['close'].iloc[i] > entry_price + 2.0 * atr_20[i] or prices['close'].iloc[i] > H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level breaks with volume and regime filters
            if vol_spike[i] and chop[i] < 61.8:  # trending market regime
                # Long signal: price crosses above L3 with volume
                if prices['close'].iloc[i] > L3_aligned[i] and prices['close'].iloc[i-1] <= L3_aligned[i-1]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short signal: price crosses below H3 with volume
                elif prices['close'].iloc[i] < H3_aligned[i] and prices['close'].iloc[i-1] >= H3_aligned[i-1]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals