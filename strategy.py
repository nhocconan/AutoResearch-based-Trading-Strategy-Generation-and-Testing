#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout from 1w with volume confirmation and chop filter.
    # Long when price breaks above Camarilla H3 level with volume spike and chop < 61.8 (trending).
    # Short when price breaks below Camarilla L3 level with volume spike and chop < 61.8.
    # Exit when price returns to Camarilla pivot point (mean reversion).
    # Uses 1w Camarilla levels aligned to 1d bars. Discrete size 0.25 to minimize fee churn.
    # Target: 30-100 total trades over 4 years (7-25/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w OHLC for Camarilla pivots
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    volume_1w = df_1w['volume'].values
    
    # Camarilla pivot levels (based on previous week)
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H3 = Close + Range * 1.1 / 4
    # L3 = Close - Range * 1.1 / 4
    # H4 = Close + Range * 1.1 / 2
    # L4 = Close - Range * 1.1 / 2
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    rng = high_1w - low_1w
    
    # H3 and L3 are the key breakout levels
    camarilla_h3 = close_1w + rng * 1.1 / 4.0
    camarilla_l3 = close_1w - rng * 1.1 / 4.0
    camarilla_pivot = pivot  # Exit level
    
    # Calculate 1w volume mean (20-period) with min_periods
    volume_series = pd.Series(volume_1w)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w choppiness index (14-period) for regime filter
    def choppiness_index(high, low, close, period=14):
        atr_sum = np.zeros_like(close)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]  # first TR
        for i in range(1, len(tr)):
            atr_sum[i] = atr_sum[i-1] + tr[i]
        atr = atr_sum / np.arange(1, len(close)+1)
        atr = np.where(np.arange(1, len(close)+1) < period, np.nan, atr_sum[period-1:] / period)
        atr_padded = np.full_like(close, np.nan)
        atr_padded[period-1:] = atr
        atr = atr_padded
        max_min_range = np.maximum.accumulate(high) - np.minimum.accumulate(low)
        chop = 100 * np.log10(atr * np.sqrt(period) / max_min_range) / np.log10(period)
        return chop
    
    chop_1w = choppiness_index(high_1w, low_1w, close_1w, 14)
    
    # Align HTF indicators to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1w volume (aligned)
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        
        # Volume filter: current 1w volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_1w_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Regime filter: chop < 61.8 (trending market)
        trending_regime = chop_aligned[i] < 61.8
        
        # Entry conditions: price breaks Camarilla H3/L3 levels with volume confirmation and trending regime
        long_entry = (close[i] > camarilla_h3_aligned[i] and volume_confirmation and trending_regime)
        short_entry = (close[i] < camarilla_l3_aligned[i] and volume_confirmation and trending_regime)
        
        # Exit conditions: price returns to Camarilla pivot point (mean reversion)
        long_exit = close[i] < camarilla_pivot_aligned[i]
        short_exit = close[i] > camarilla_pivot_aligned[i]
        
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

name = "1d_1w_camarilla_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0