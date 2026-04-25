#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_RegimeFilter_v22
Hypothesis: Camarilla R1/S1 breakout with 1d EMA50 trend filter and ADX/chop regime filter reduces overtrading while capturing institutional moves. Uses discrete sizing (0.25) and ATR stoploss (2.0) with 8-bar minimum hold. Regime filter uses ADX>25 for trending markets (breakouts) and ADX<20 for ranging markets (mean reversion at Camarilla H3/L3 levels). Works in bull/bear by adapting to market regime.
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
    
    # Get 1d data for trend filter (EMA50) and regime (ADX) - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need for EMA50 and ADX
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR for stoploss calculation
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d ADX for regime filter (trending vs ranging)
    # ADX calculation: +DI, -DI, DX, then smoothed
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=1).sum().values
    tr_14 = np.concatenate([[np.nan] * 13, tr_14[13:]])  # align with tr
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h data for Camarilla calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h
    # Camarilla levels based on previous bar's range
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r1 = prev_close + 0.5 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 0.5 * (prev_high - prev_low)
    camarilla_r3 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.5 * (prev_high - prev_low)
    camarilla_h3 = camarilla_pp + 1.5 * (high_4h - low_4h)  # H3 for ranging longs
    camarilla_l3 = camarilla_pp - 1.5 * (high_4h - low_4h)  # L3 for ranging shorts
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Volume filter: volume > 2.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14, 14)  # EMA50 needs 50, vol MA needs 20, ATR needs 14, ADX needs 14+14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        h3_val = camarilla_h3_aligned[i]
        l3_val = camarilla_l3_aligned[i]
        
        # Get 4h close aligned for direct comparison
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
        close_4h_val = close_4h_aligned[i]
        is_uptrend = close_4h_val > ema_50_val
        
        # Regime determination
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # Look for entry signals based on regime
            long_signal = False
            short_signal = False
            
            if is_trending:
                # Trending market: breakout in direction of 1d trend
                long_signal = (close_4h_val > r1_val) and is_uptrend and vol_spike[i]
                short_signal = (close_4h_val < s1_val) and (not is_uptrend) and vol_spike[i]
            elif is_ranging:
                # Ranging market: mean reversion at extreme levels
                long_signal = (close_4h_val < s3_val) and vol_spike[i]  # oversold bounce
                short_signal = (close_4h_val > r3_val) and vol_spike[i]  # overbought rejection
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_4h_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_4h_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit conditions based on regime
            if bars_since_entry >= 8:
                if is_trending:
                    # Trending: exit on opposite Camarilla level or ATR stop
                    exit_signal = close_4h_val < s1_val
                    stop_signal = close_4h_val < (entry_price - 2.0 * atr_val)
                else:
                    # Ranging: exit at mean reversion target or ATR stop
                    exit_signal = close_4h_val > h3_val
                    stop_signal = close_4h_val < (entry_price - 2.0 * atr_val)
                if exit_signal or stop_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit conditions based on regime
            if bars_since_entry >= 8:
                if is_trending:
                    # Trending: exit on opposite Camarilla level or ATR stop
                    exit_signal = close_4h_val > r1_val
                    stop_signal = close_4h_val > (entry_price + 2.0 * atr_val)
                else:
                    # Ranging: exit at mean reversion target or ATR stop
                    exit_signal = close_4h_val < l3_val
                    stop_signal = close_4h_val > (entry_price + 2.0 * atr_val)
                if exit_signal or stop_signal:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_RegimeFilter_v22"
timeframe = "4h"
leverage = 1.0