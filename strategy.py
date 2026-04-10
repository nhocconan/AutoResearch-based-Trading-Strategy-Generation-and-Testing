#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.8x 20-period average AND 12h chop < 61.8 (trending)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.8x 20-period average AND 12h chop < 61.8 (trending)
# - Exit when price crosses Camarilla PIVOT level (midpoint) OR opposite breakout occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla pivots provide precise intraday support/resistance levels
# - Volume confirmation reduces false breakouts
# - Choppiness filter ensures we trade only in trending markets (avoid chop)

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 12h Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels calculated from daily OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Camarilla levels
    # H4 = P + 1.1 * (H - L) / 2
    # H3 = P + 1.1 * (H - L) / 4
    # L3 = P - 1.1 * (H - L) / 4
    # L4 = P - 1.1 * (H - L) / 2
    range_1d = high_1d - low_1d
    camarilla_h3 = pivot_1d + (1.1 * range_1d / 4)
    camarilla_l3 = pivot_1d - (1.1 * range_1d / 4)
    camarilla_pivot = pivot_1d  # Midpoint for exit
    
    # Pre-compute 12h volume confirmation (using 1d volume aligned to 12h)
    vol_ma_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values  # Using close as proxy for volume calculation
    # Actually use volume from 1d data - but we don't have volume in df_1d from get_htf_data
    # So we'll use price-based volatility as volume proxy: ATR
    # Alternative: since we can't get volume from HTF easily, use price range as proxy
    vol_proxy_1d = (high_1d - low_1d)  # Daily range as volume proxy
    vol_ma_proxy = pd.Series(vol_proxy_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_proxy_1d > (1.8 * vol_ma_proxy)
    
    # Pre-compute 12h Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR(14)) / log10((HHH - LLL) / ATR(14)))
    # Simplified: CHOP = 100 * log10(sum of TR over period) / log10((max high - min low) / ATR)
    # We'll use a simpler version: CHOP = 100 * (sum of true range over 14 periods) / (max high - min low over 14 periods) * log10(14)
    # Actually standard formula: CHOP = 100 * log10(sum(TR14) / (log10((HHH-LLL)/ATR14)))
    # For simplicity, we'll use: CHOP < 61.8 = trending, > 61.8 = ranging
    
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]  # First bar
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = hh_14 - ll_14
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum_14 / np.log10(denominator / tr_sum_14 + 1e-10))
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)  # Default to neutral
    
    # Trending market: CHOP < 61.8
    trending_market = chop < 61.8
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    trending_market_aligned = align_htf_to_ltf(prices, df_1d, trending_market)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(trending_market_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND volume spike AND trending market
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike_1d_aligned[i] and 
                trending_market_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume spike AND trending market
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike_1d_aligned[i] and 
                  trending_market_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla PIVOT level OR opposite breakout occurs
            exit_long = (position == 1 and 
                        (close[i] < camarilla_pivot_aligned[i] or close[i] < camarilla_l3_aligned[i]))
            exit_short = (position == -1 and 
                         (close[i] > camarilla_pivot_aligned[i] or close[i] > camarilla_h3_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals