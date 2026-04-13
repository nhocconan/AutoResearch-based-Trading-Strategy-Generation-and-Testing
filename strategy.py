#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
    # Long: price breaks above H3 pivot AND volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
    # Short: price breaks below L3 pivot AND volume > 1.5x 20-bar avg AND chop < 61.8 (trending)
    # Exit: price returns to Pivot Point or opposite pivot break
    # Using 4h primary for optimal trade frequency, Camarilla for structure,
    # 1d volume for confirmation, chop filter to avoid whipsaws in ranging markets.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume confirmation and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    # But we need previous day's OHLC for today's levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    H3 = pivot + 1.1 * range_hl
    L3 = pivot - 1.1 * range_hl
    H4 = pivot + 1.5 * range_hl
    L4 = pivot - 1.5 * range_hl
    PP = pivot  # Pivot Point
    
    # Align daily Camarilla levels to 4h (already aligned to completed day by get_htf_data)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    PP_4h = align_htf_to_ltf(prices, df_1d, PP)
    
    # Calculate daily volume for confirmation (>1.5x 20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    volume_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    volume_spike_4h = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 4h Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR1) / (n * log(n))) / log10(n)
    # Where ATR1 = True Range, n = period
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    tr = true_range(high, low, close_prev)
    
    atr_period = 14
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.mean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Sum of ATR over CHOP period
    chop_period = 14
    sum_atr = np.full(n, np.nan)
    for i in range(chop_period, n):
        sum_atr[i] = np.sum(atr[i-chop_period+1:i+1])
    
    # Calculate CHOP
    chop = np.full(n, np.nan)
    for i in range(chop_period, n):
        if sum_atr[i] > 0 and i > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (chop_period * np.log10(chop_period))) / np.log10(chop_period)
    
    # Trending regime: CHOP < 61.8
    trending_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or 
            np.isnan(volume_spike_4h[i]) or np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike_4h[i]
        
        # Regime filter: trending market
        regime_filter = trending_regime[i]
        
        # Entry logic: Camarilla breakout + volume + regime
        long_entry = (close[i] > H3_4h[i]) and vol_confirm and regime_filter
        short_entry = (close[i] < L3_4h[i]) and vol_confirm and regime_filter
        
        # Exit logic: return to pivot or opposite break
        long_exit = (close[i] < PP_4h[i]) or (close[i] > H4_4h[i])
        short_exit = (close[i] > PP_4h[i]) or (close[i] < L4_4h[i])
        
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

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0