#!/usr/bin/env python3
# Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; in ranging markets (CHOP > 50) we fade extremes.
# In trending markets (CHOP <= 50) we follow the 1d EMA50 direction.
# Volume spike (>1.5x average) confirms conviction. Uses discrete sizing (0.25) to minimize fee churn.
# Designed for 12h timeframe to keep trades ~12-37/year, avoiding overtrading.
# Works in bull markets via trend-following pullsbacks and in bear markets via mean reversion in ranges.

name = "12h_WilliamsR_1dTrend_VolumeChop_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation (14-period)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R on 12h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_12h) / (highest_high - lowest_low)) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Choppiness Index regime filter (14-period) on 12h
    # CHOP = 100 * log10(sum(ATR) / (log10(highest high - lowest low) * n)) / log10(n)
    tr1 = pd.Series(high_12h).rolling(window=14, min_periods=1).max() - pd.Series(low_12h).rolling(window=14, min_periods=1).min()
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(highest_high_14 - lowest_low_14) * 14
    chop = np.where(chop_denominator > 0, 100 * np.log10(atr_sum) / chop_denominator, 50)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Align 1d and 12h indicators to 12h timeframe (identity for 12h, aligned for 1d)
    williams_r_aligned = williams_r  # already 12h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_aligned = volume_filter  # volume is already 12h aligned
    chop_aligned = chop  # chop is already 12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_filter_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ok = volume_filter_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # In ranging market (CHOP > 50): mean reversion from extremes
            if chop_val > 50:
                # Long when oversold (%R < -80) with volume confirmation
                if wr < -80 and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short when overbought (%R > -20) with volume confirmation
                elif wr > -20 and vol_ok:
                    signals[i] = -0.25
                    position = -1
            # In trending market (CHOP <= 50): follow EMA50 direction on pullbacks
            else:
                # Long: price above EMA50 and pulling back to oversold (%R < -50)
                if close[i] > ema_trend and wr < -50 and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short: price below EMA50 and pulling back to overbought (%R > -50)
                elif close[i] < ema_trend and wr > -50 and vol_ok:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: %R reaches overbought (-20) or trend reversal (price < EMA50 in trending market)
            if wr > -20 or (chop_val <= 50 and close[i] < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: %R reaches oversold (-80) or trend reversal (price > EMA50 in trending market)
            if wr < -80 or (chop_val <= 50 and close[i] > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals