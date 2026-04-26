#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: On 4h timeframe, Donchian(20) breakouts filtered by 1d EMA50 trend and volume spike capture institutional moves with low trade frequency. Long when price breaks above upper Donchian in bullish 1d trend with volume confirmation; short when breaks below lower Donchian in bearish 1d trend. Uses discrete sizing (±0.25) and ATR-based stoploss. Designed for BTC/ETH with regime filter (ADX>25) to avoid whipsaws in ranging markets. Targets 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR for stoploss and regime filter
    tr1 = pd.Series(high).rolling(window=1).max() - pd.Series(low).rolling(window=1).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ADX for regime filter (trending market)
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_di_14 = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / tr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / tr_14)
    dx = 100 * abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of calculations (20 for Donchian, 14 for ATR/ADX, 20 for volume MA)
    start_idx = max(20, 14, 20) + 4  # +4 to ensure 1d bar completion (4h -> 1d: 6 bars per 1d)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        adx_val = adx[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending_market = adx_val > 25
        
        # Entry conditions: price breaks Donchian levels in direction of 1d trend with volume confirmation and trending regime
        long_entry = (close_val > highest_high[i]) and bullish_1d and vol_spike and trending_market
        short_entry = (close_val < lowest_low[i]) and bearish_1d and vol_spike and trending_market
        
        # Stoploss: ATR-based (2x ATR from entry)
        if position == 1 and i > start_idx:
            # Approximate entry price as the breakout level
            entry_price = highest_high[i-1] if close_val > highest_high[i-1] else close_val
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1 and i > start_idx:
            entry_price = lowest_low[i-1] if close_val < lowest_low[i-1] else close_val
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # Exit conditions: price returns inside Donchian channels or trend reversal
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < lowest_low[i] or not bullish_1d):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > highest_high[i] or not bearish_1d):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0