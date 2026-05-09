# 12h_Camarilla_Pivot_Scalper_v1
# Hypothesis: Camarilla pivot levels from 1d provide high-probability reversal zones. 
# In sideways markets (chop > 61.8), price tends to revert to pivot (H5/L5). 
# In trending markets (chop < 38.2), breakouts of H4/L4 with volume continue the trend.
# Uses 1d Camarilla + 12h chop filter + volume confirmation for entries.
# Targets 15-25 trades/year to minimize fee drag while capturing reversals and breakouts.
# Works in bull/bear via regime adaptation: mean reversion in chop, trend following in trends.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_Scalper_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Chop Index (14-period) for regime detection
    tr1 = high[1:] - low[:-1]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[:-1] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14.sum() / (hh14 - ll14)) / np.log10(14) if False else np.zeros(n)  # placeholder
    # Calculate chop properly: need sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_range = hh14 - ll14
    chop = 100 * np.log10(tr_sum / hh_ll_range) / np.log10(14)
    chop = np.where(hh_ll_range > 0, chop, 50)  # avoid div by zero
    
    # Daily Camarilla levels (from previous day)
    # Need previous day's OHLC - we'll use daily data shifted by 1
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # True range for Camarilla (using previous day)
    tr_camarilla = np.maximum(prev_high - prev_low, 
                              np.maximum(np.abs(prev_high - prev_close), 
                                         np.abs(prev_low - prev_close)))
    
    # Camarilla levels
    H4 = prev_close + 1.1 * tr_camarilla / 2
    L4 = prev_close - 1.1 * tr_camarilla / 2
    H5 = prev_close + 1.1 * tr_camarilla
    L5 = prev_close - 1.1 * tr_camarilla
    H3 = prev_close + 1.1 * tr_camarilla / 4
    L3 = prev_close - 1.1 * tr_camarilla / 4
    
    # Align Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H5_aligned = align_htf_to_ltf(prices, df_1d, H5)
    L5_aligned = align_htf_to_ltf(prices, df_1d, L5)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(chop[i]) or np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(H5_aligned[i]) or np.isnan(L5_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Choppy market (chop > 61.8): mean reversion to H5/L5
            if chop[i] > 61.8:
                # Long near L5 with volume
                if price <= L5_aligned[i] * 1.002 and price >= L5_aligned[i] * 0.998 and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short near H5 with volume
                elif price >= H5_aligned[i] * 0.998 and price <= H5_aligned[i] * 1.002 and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    continue
            # Trending market (chop < 38.2): breakout of H4/L4 with volume
            elif chop[i] < 38.2:
                # Long breakout above H4
                if price > H4_aligned[i] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short breakdown below L4
                elif price < L4_aligned[i] and vol_ok:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        elif position == 1:
            # Exit long: chop increases (range-bound) or price reaches H3 (take profit)
            if chop[i] > 50 or price >= H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: chop increases or price reaches L3
            if chop[i] > 50 or price <= L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals