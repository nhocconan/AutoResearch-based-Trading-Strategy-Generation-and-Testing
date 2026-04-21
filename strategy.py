#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_And_Chop_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 1d for trend direction, 
filtered by choppiness index (range/trend regime) and volume confirmation on 1d.
Works in bull/bear: KAMA adapts to market noise, chop filter avoids whipsaws in ranging markets,
volume ensures institutional participation. Weekly trend from 1w EMA200 for long-term bias.
Target: 15-25 trades/year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for KAMA, chop, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for KAMA and chop
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) on 1d ===
    # ER (Efficiency Ratio) = |net change| / sum of absolute changes
    # Smooth constant = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # where fastest SC = 2/(2+1) = 0.6667, slowest SC = 2/(30+1) = 0.0645
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    abs_change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    
    # 10-period ER
    net_change = np.abs(np.subtract(close_1d, np.roll(close_1d, 10)))
    net_change[:10] = np.nan
    vol_sum = np.nancumsum(abs_change) - np.nancumsum(np.roll(abs_change, 10))
    vol_sum[:10] = np.nan
    er = np.where(vol_sum != 0, net_change / vol_sum, 0)
    
    # Smooth constant
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === Choppiness Index on 1d (14-period) ===
    # Chop = 100 * log10(sum(TR) / (HHV - LLV)) / log10(14)
    # where TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # 14-period sums and ranges
    tr_sum = np.nancumsum(tr) - np.nancumsum(np.roll(tr, 14))
    tr_sum[:14] = np.nan
    hh = np.maximum.accumulate(high_1d)
    ll = np.minimum.accumulate(low_1d)
    hh_14 = hh - np.roll(hh, 14)
    ll_14 = ll - np.roll(ll, 14)
    range_14 = np.maximum(hh_14, ll_14)
    range_14[:14] = np.nan
    
    chop = 100 * np.log10(tr_sum / range_14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Volume confirmation on 1d ===
    # Volume > 1.5 * 20-period average
    vol_ma = np.nancumsum(volume_1d) - np.nancumsum(np.roll(volume_1d, 20))
    vol_ma[:20] = np.nan
    vol_ma = vol_ma / 20
    volume_ok = volume_1d > (1.5 * vol_ma)
    volume_ok_aligned = align_htf_to_ltf(prices, df_1d, volume_ok.astype(float))
    
    # === Weekly trend bias from 1w EMA200 ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ok_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Regime filter: chop < 61.8 = trending (favor trend following), chop > 61.8 = ranging (avoid)
        # We'll use chop < 61.8 to ensure we're in a trending environment
        is_trending = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price > KAMA AND trending regime AND volume OK AND weekly bullish (price > weekly EMA200)
            if (price > kama_aligned[i] and 
                is_trending and 
                volume_ok_aligned[i] and 
                price > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND trending regime AND volume OK AND weekly bearish (price < weekly EMA200)
            elif (price < kama_aligned[i] and 
                  is_trending and 
                  volume_ok_aligned[i] and 
                  price < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < KAMA (trend reversal) OR chop > 61.8 (entered ranging market)
            if price < kama_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA (trend reversal) OR chop > 61.8 (entered ranging market)
            if price > kama_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_Volume_And_Chop_Filter"
timeframe = "1d"
leverage = 1.0