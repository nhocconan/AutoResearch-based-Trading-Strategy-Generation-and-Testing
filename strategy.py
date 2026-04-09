#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d chop regime filter
# Camarilla pivots provide precise intraday S/R levels for breakout/mean reversion
# 4h volume spike confirms institutional participation (avoids false breakouts)
# 1d chop regime filter: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert at H3/L3)
# Works in bull/bear: regime filter adapts, Camarilla levels tighten in volatility
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20
# Timeframe: 1h (primary), HTF: 4h (volume), 1d (chop)

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h average volume (20-period)
    volume_4h = df_4h['volume'].values
    volume_s_4h = pd.Series(volume_4h)
    avg_volume_4h = volume_s_4h.rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(atr14) / (hh14 - ll14)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)  # neutral when range is zero
    
    # Align HTF indicators to 1h timeframe
    avg_volume_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_volume_4h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 1h Camarilla levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # H4 = Pivot + 1.1 * (H - L) / 2
    # L4 = Pivot - 1.1 * (H - L) / 2
    # H3 = Pivot + 1.1 * (H - L) / 4
    # L3 = Pivot - 1.1 * (H - L) / 4
    # H2 = Pivot + 1.1 * (H - L) / 6
    # L2 = Pivot - 1.1 * (H - L) / 6
    # H1 = Pivot + 1.1 * (H - L) / 12
    # L1 = Pivot - 1.1 * (H - L) / 12
    
    # Use daily OHLC from 1d timeframe for Camarilla calculation
    # We need to align daily OHLC to 1h bars
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily OHLC to 1h (wait for daily bar close)
    daily_open_aligned = align_htf_to_ltf(prices, df_1d, daily_open)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate Camarilla levels for each 1h bar using aligned daily OHLC
    pivot = (daily_high_aligned + daily_low_aligned + daily_close_aligned) / 3.0
    hl_range = daily_high_aligned - daily_low_aligned
    
    H4 = pivot + 1.1 * hl_range / 2.0
    L4 = pivot - 1.1 * hl_range / 2.0
    H3 = pivot + 1.1 * hl_range / 4.0
    L3 = pivot - 1.1 * hl_range / 4.0
    H2 = pivot + 1.1 * hl_range / 6.0
    L2 = pivot - 1.1 * hl_range / 6.0
    H1 = pivot + 1.1 * hl_range / 12.0
    L1 = pivot - 1.1 * hl_range / 12.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(H4[i]) or np.isnan(L4[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or
            np.isnan(avg_volume_4h_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.8x 4h average volume
        volume_confirmed = volume[i] > 1.8 * avg_volume_4h_aligned[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow breakout), CHOP > 61.8 = range (mean revert)
        trending_regime = chop_1d_aligned[i] < 38.2
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR regime shifts to ranging
            if close[i] < L3[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR regime shifts to ranging
            if close[i] > H3[i] or ranging_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic
            if trending_regime:
                # Follow breakout in trending regime
                if close[i] > H4[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.20
                elif close[i] < L4[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.20
            elif ranging_regime:
                # Mean revert at H3/L3 in ranging regime
                if close[i] < L3[i] and volume_confirmed:
                    position = 1
                    signals[i] = 0.20
                elif close[i] > H3[i] and volume_confirmed:
                    position = -1
                    signals[i] = -0.20
    
    return signals