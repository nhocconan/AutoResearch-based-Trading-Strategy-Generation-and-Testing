#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wEMA34_VolumeSpike_ChopRegime
Hypothesis: Camarilla R1/S1 breakout on 12h timeframe with 1w EMA34 trend filter, volume spike confirmation (>2.0x average), and choppiness regime filter (CHOP > 61.8 for mean reversion, < 38.2 for trend following). Only trade breakouts aligned with 1w EMA34 direction during high volume expansion and appropriate market regime. Uses discrete sizing (0.25) and ATR-based stoploss. Designed to capture strong momentum moves in both bull and bear markets by using the 1w EMA as a dynamic trend filter and chop filter to avoid whipsaws in ranging markets. Targets 12-37 trades/year on 12h timeframe to minimize fee drag.
"""

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
    
    # Get 12h data for Camarilla levels and ATR - primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate previous period's Camarilla levels (using prior 12h bar's HLC)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for R1, R2, S1, S2
    rang = prev_high - prev_low
    r2 = prev_close + (rang * 1.1 / 2)
    r1 = prev_close + (rang * 1.1 / 4)
    s1 = prev_close - (rang * 1.1 / 4)
    s2 = prev_close - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # Get 1w data for EMA34 trend filter - HTF
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate ATR(14) for stoploss on 12h
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (CHOP) on 12h for regime filter
    # CHOP = 100 * LOG10(SUM(ATR(1), n) / (LOG10(HHIGH - LLOW) / LOG10(2))) / LOG10(n)
    # Simplified: CHOP = 100 * LOG10(ATR_sum / (LOG10(range) / LOG10(2))) / LOG10(period)
    atr_1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_1[0] = high[0] - low[0]  # first bar
    
    atr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    
    # Avoid division by zero and log of zero
    log_range = np.log10(np.maximum(range_14, 1e-10))
    log_atr_sum = np.log10(np.maximum(atr_sum, 1e-10))
    chop = 100 * (log_atr_sum / (log_range / np.log10(2))) / np.log10(14)
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 20, 14, 14)  # EMA needs 34, vol needs 20, ATR needs 14, CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        r2_val = r2_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        s2_val = s2_aligned[i]
        ema_val = ema_1w_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        chop_val = chop[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        # Regime filters
        is_trending = chop_val < 38.2  # Trending regime
        is_ranging = chop_val > 61.8   # Ranging regime
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend and volume confirmation
            # In trending regime: breakout in direction of trend
            # In ranging regime: mean reversion at extremes
            long_signal = False
            short_signal = False
            
            if is_trending:
                # Trending: breakout with trend
                long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike
                short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike
            elif is_ranging:
                # Ranging: mean reversion at support/resistance
                long_signal = (low_val <= s1_val) and (close_val > s1_val) and volume_spike  # bounce off S1
                short_signal = (high_val >= r1_val) and (close_val < r1_val) and volume_spike  # rejection at R1
            
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
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.5*ATR
            if close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes below 1w EMA34 (in trending) or at S1 (in ranging)
            elif is_trending and close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif is_ranging and close_val <= s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.5*ATR
            if close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes above 1w EMA34 (in trending) or at R1 (in ranging)
            elif is_trending and close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif is_ranging and close_val >= r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wEMA34_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0