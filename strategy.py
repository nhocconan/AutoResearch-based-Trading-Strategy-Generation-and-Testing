#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with 1d volume spike and choppiness regime filter.
# TRIX (12) captures smooth momentum; long when TRIX > 0 and rising, short when TRIX < 0 and falling.
# Volume confirmation: current 4h volume > 2.0x 20-period median to avoid low-volume breakouts.
# Choppiness regime: CHOP(14) > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow).
# In ranging markets, fade TRIX extremes; in trending markets, follow TRIX direction.
# Discrete position sizing (0.25) to minimize fee churn. Target: 80-180 trades over 4 years.

name = "4h_TRIX_Volume_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter (used in regime logic)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for choppiness calculation
    tr1 = np.maximum(df_1d['high'].values[1:] - df_1d['low'].values[1:], 
                     np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1]))
    tr2 = np.maximum(np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1]), 
                     np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d True Range sum and high-low range for Choppiness
    atr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum_14 / (hh_14 - ll_14)) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Calculate TRIX(12,9) on 4h close
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = np.nan  # first value is invalid
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for TRIX and volume median
    start_idx = max(34, 20) + 1  # 35
    
    for i in range(start_idx, n):
        if (np.isnan(trix_raw[i]) or 
            np.isnan(trix_signal[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # TRIX momentum: TRIX > signal line = bullish momentum
        trix_bullish = trix_raw[i] > trix_signal[i]
        trix_bearish = trix_raw[i] < trix_signal[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Choppiness regime: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
        chop_value = chop_aligned[i]
        ranging_market = chop_value > 61.8
        trending_market = chop_value < 38.2
        
        if position == 0:  # Flat - look for new entries
            # In trending market: follow TRIX momentum
            if trending_market:
                # Long: TRIX bullish AND volume confirmation
                if trix_bullish and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX bearish AND volume confirmation
                elif trix_bearish and volume_confirm:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # In ranging market: fade TRIX extremes (mean reversion)
            elif ranging_market:
                # Long: TRIX oversold (below -0.15) AND volume confirmation
                if trix_raw[i] < -0.15 and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX overbought (above +0.15) AND volume confirmation
                elif trix_raw[i] > 0.15 and volume_confirm:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            # In choppy transition zone: no trades
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if trending_market:
                # Exit long when TRIX turns bearish
                if trix_bearish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif ranging_market:
                # Exit long when TRIX reaches neutral zone
                if trix_raw[i] > -0.05:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            if trending_market:
                # Exit short when TRIX turns bullish
                if trix_bullish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif ranging_market:
                # Exit short when TRIX reaches neutral zone
                if trix_raw[i] < 0.05:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals