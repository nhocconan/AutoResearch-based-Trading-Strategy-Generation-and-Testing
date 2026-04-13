#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
    # Long: price breaks above H3 (1d) AND volume > 1.5x 20-period avg AND CHOP(14) < 61.8 (trending)
    # Short: price breaks below L3 (1d) AND volume > 1.5x 20-period avg AND CHOP(14) < 61.8 (trending)
    # Exit: price retreats to Pivot (1d) or volume dry-up
    # Using 12h timeframe for low trade frequency, Camarilla for institutional levels,
    # volume for confirmation, chop regime to avoid ranging markets.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.7*(high-low)
    # H1 = close + 0.5*(high-low)
    # Pivot = (high+low+close)/3
    # L1 = close - 0.5*(high-low)
    # L2 = close - 0.7*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    rng = high_1d - low_1d
    H3 = close_1d + 1.1 * rng
    L3 = close_1d - 1.1 * rng
    Pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align to 12h timeframe
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    Pivot_12h = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # Calculate daily Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range)) / log10(14)
    # High CHOP (>61.8) = ranging, Low CHOP (<38.2) = trending
    
    # True Range calculation
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(np.diff(low_1d))
    tr3 = np.abs(np.diff(close_1d))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    # ATR(14)
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.nanmean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR(14) over last 14 periods
    atr_sum = np.full(len(tr), np.nan)
    for i in range(27, len(tr)):  # 14+13
        atr_sum[i] = np.nansum(atr_14[i-13:i+1])
    
    # Rolling max/min of high/low over 14 periods
    max_high = np.full(len(tr), np.nan)
    min_low = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        max_high[i] = np.nanmax(high_1d[i-13:i+1])
        min_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Chopiness Index
    chop = np.full(len(tr), np.nan)
    for i in range(27, len(tr)):
        if max_high[i] > min_low[i] and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral
    
    # Align chop to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get daily volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or 
            np.isnan(Pivot_12h[i]) or np.isnan(chop_12h[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: CHOP < 61.8 = trending (favor breakouts)
        trending_regime = chop_12h[i] < 61.8
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Camarilla breakout + volume + regime
        long_entry = (close[i] > H3_12h[i]) and vol_confirm and trending_regime
        short_entry = (close[i] < L3_12h[i]) and vol_confirm and trending_regime
        
        # Exit logic: retreat to pivot or volume dry-up
        long_exit = (close[i] < Pivot_12h[i]) or not vol_confirm
        short_exit = (close[i] > Pivot_12h[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0