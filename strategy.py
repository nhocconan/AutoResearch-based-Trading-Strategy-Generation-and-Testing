#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_ATRStop
Hypothesis: Trade 4h TRIX zero-line crosses with volume confirmation and choppiness regime filter.
TRIX (12,20,9) filters noise and captures momentum. Volume spike confirms institutional participation.
Choppiness regime (CHOP > 61.8) ensures range-bound conditions for mean-reversion exits.
ATR-based stoploss controls risk. Works in bull/bear by adapting to regime: trend follow in trending (CHOP < 38.2),
mean-revert in choppy (CHOP > 61.8). Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate TRIX on 4h: EMA(EMA(EMA(close,12),20),9) - percentage change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Choppiness Index on 1d: CHOP = 100 * log10(sum(ATR(14),14) / (max(high,14)-min(low,14))) / log10(14)
    # Simplified: use true range and range over 14 periods
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - np.concatenate([[np.nan], df_1d['close'].values[:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[np.nan], df_1d['close'].values[:-1]]))))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of TRIX warmup, EMA, ATR, volume MA, CHOP
    start_idx = max(12+20+9, 34, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trix_val = trix[i]
        trix_prev = trix[i-1]
        trend_1d_up = close_val > ema_34_1d_aligned[i]
        trend_1d_down = close_val < ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop_aligned[i]
        
        # Regime filters
        is_choppy = chop_val > 61.8   # range-bound: mean revert
        is_trending = chop_val < 38.2  # trending: trend follow
        
        if position == 0:
            # Entry logic adapts to regime
            if is_choppy:
                # In choppy market: mean reversion at TRIX extremes
                long_signal = (trix_prev <= -0.5 and trix_val > -0.5) and vol_spike  # TRIX crosses up from oversold
                short_signal = (trix_prev >= 0.5 and trix_val < 0.5) and vol_spike   # TRIX crosses down from overbought
            elif is_trending:
                # In trending market: trend follow with TRIX zero-line cross
                long_signal = (trix_prev <= 0 and trix_val > 0) and trend_1d_up and vol_spike
                short_signal = (trix_prev >= 0 and trix_val < 0) and trend_1d_down and vol_spike
            else:
                # Transition regime: require stronger signal
                long_signal = (trix_prev <= 0 and trix_val > 0.2) and trend_1d_up and vol_spike
                short_signal = (trix_prev >= 0 and trix_val < -0.2) and trend_1d_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit logic adapts to regime
            if is_choppy:
                # In choppy: exit at TRIX overbought or stoploss
                if (trix_val >= 0.5) or (close_val < entry_price - 1.5 * atr[i]):
                    signals[i] = 0.0
                    position = 0
            else:
                # In trending: exit at trend flip or stoploss
                if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]):
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit logic adapts to regime
            if is_choppy:
                # In choppy: exit at TRIX oversold or stoploss
                if (trix_val <= -0.5) or (close_val > entry_price + 1.5 * atr[i]):
                    signals[i] = 0.0
                    position = 0
            else:
                # In trending: exit at trend flip or stoploss
                if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]):
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_ATRStop"
timeframe = "4h"
leverage = 1.0