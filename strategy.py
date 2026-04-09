#!/usr/bin/env python3
# 12h_camarilla_1d_trend_volume_v4
# Hypothesis: 12h Camarilla pivot levels from 1d HTF + volume spike + chop regime filter.
# Uses 1d Camarilla levels (H3/L3) for entry/exit with 1d EMA50 trend filter.
# Volume confirmation avoids false breakouts. Chop filter (34) avoids whipsaws in sideways markets.
# Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.
# Works in bull/bear: trend filter captures institutional direction, Camarilla provides mean-reversion
# levels that work in ranging markets, volume confirms participation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_trend_volume_v4"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d Camarilla levels (H3, L3, H4, L4)
    # H3 = high + 1.1*(high-low)/2, L3 = low - 1.1*(high-low)/2
    # H4 = high + 1.1*(high-low), L4 = low - 1.1*(high-low)
    hl_range_1d = high_1d - low_1d
    H3_1d = high_1d + 1.1 * hl_range_1d / 2
    L3_1d = low_1d - 1.1 * hl_range_1d / 2
    H4_1d = high_1d + 1.1 * hl_range_1d
    L4_1d = low_1d - 1.1 * hl_range_1d
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (34-period) - avoids whipsaws
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    atr_period = 34
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # True range sum and price range for chop calculation
    tr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    price_range = max_high - min_low
    
    # Avoid division by zero
    chop = np.zeros_like(tr_sum)
    mask = price_range > 0
    chop[mask] = 100 * np.log10(tr_sum[mask] / price_range[mask]) / np.log10(atr_period)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or
            np.isnan(H4_1d_aligned[i]) or np.isnan(L4_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 OR trend turns bearish
            if close[i] < L3_1d_aligned[i] or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above H3 OR trend turns bullish
            if close[i] > H3_1d_aligned[i] or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and chop filter
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            chop_filter = chop[i] > 50.0  # Avoid strong trending regimes for mean reversion
            
            if volume_confirmed and chop_filter:
                # Long: price crosses above L3 with bullish trend
                if close[i] > L3_1d_aligned[i] and close[i] > ema50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below H3 with bearish trend
                elif close[i] < H3_1d_aligned[i] and close[i] < ema50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals