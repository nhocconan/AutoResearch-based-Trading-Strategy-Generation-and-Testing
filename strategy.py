#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX mean reversion with 1d volume spike and choppiness regime filter
# - TRIX(12) crossing zero line indicates momentum shift
# - Long when TRIX crosses above zero AND 1d volume > 2.0x 20-bar avg AND 1d chop > 61.8 (range regime)
# - Short when TRIX crosses below zero AND 1d volume > 2.0x 20-bar avg AND 1d chop > 61.8
# - Exit when TRIX returns to zero (mean reversion) or chop < 38.2 (trend regime)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - TRIX filters noise better than MACD; volume spike confirms institutional interest
# - Choppiness regime filter ensures mean reversion only in ranging markets
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: mean reversion in ranges, avoids trend markets

name = "12h_1d_trix_meanreversion_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d TRIX(12): triple EMA of close, then ROC
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, min_periods=12, adjust=False).mean().values
    # TRIX = 100 * (EMA3 today - EMA3 yesterday) / EMA3 yesterday
    trix_raw = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix = np.concatenate([[np.nan], trix_raw])  # align with original length
    # TRIX signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix).ewm(span=9, min_periods=9, adjust=False).mean().values
    # TRIX histogram = TRIX - signal
    trix_hist = trix - trix_signal
    
    # TRIX zero cross signals
    trix_cross_above = (trix_hist[:-1] <= 0) & (trix_hist[1:] > 0)  # bullish cross
    trix_cross_below = (trix_hist[:-1] >= 0) & (trix_hist[1:] < 0)  # bearish cross
    # Shift to align with bar where cross completed
    trix_cross_above = np.concatenate([[False], trix_cross_above])
    trix_cross_below = np.concatenate([[False], trix_cross_below])
    # Exit when TRIX returns near zero
    trix_exit = np.abs(trix_hist) < 0.05
    
    # Align 1d TRIX signals to 12h timeframe
    trix_cross_above_aligned = align_htf_to_ltf(prices, df_1d, trix_cross_above)
    trix_cross_below_aligned = align_htf_to_ltf(prices, df_1d, trix_cross_below)
    trix_exit_aligned = align_htf_to_ltf(prices, df_1d, trix_exit)
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * volume_20_avg)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 1d Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (HH - LL)))
    # Simplified: CHOP = 100 * (sum of true range over period) / (ATR(1) * period) normalized
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[0], tr])  # first TR = 0
    # ATR(14) = EMA of TR
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    # Highest High and Lowest Low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum_atr_14 / (log10(14) * (hh_14 - ll_14))) / log10(14)
    # Avoid division by zero
    hl_range = hh_14 - ll_14
    chop_raw = 100 * np.log10(sum_atr_14 / (np.log10(14) * hl_range)) / np.log10(14)
    chop = np.where((hl_range == 0) | np.isnan(sum_atr_14) | np.isnan(hl_range), 50, chop_raw)
    chop = np.nan_to_num(chop, nan=50.0)
    # Regime filters: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
    chop_range = chop > 61.8
    chop_trend = chop < 38.2
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range)
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(trix_cross_above_aligned[i]) or np.isnan(trix_cross_below_aligned[i]) or
            np.isnan(trix_exit_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(chop_range_aligned[i]) or np.isnan(chop_trend_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries in range regime
            # Long when TRIX bullish cross AND volume spike AND range regime
            if (trix_cross_above_aligned[i] and 
                vol_spike_aligned[i] and 
                chop_range_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when TRIX bearish cross AND volume spike AND range regime
            elif (trix_cross_below_aligned[i] and 
                  vol_spike_aligned[i] and 
                  chop_range_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when TRIX returns to zero OR regime changes to trending
            exit_signal = trix_exit_aligned[i] or chop_trend_aligned[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals