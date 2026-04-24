#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ATR trend filter + volume confirmation + ATR stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(14) for trend filter (ATR rising = trending market, ATR falling = ranging).
- Entry: Long when close breaks above Donchian upper(20) AND 1d ATR(14) > its 20-period MA AND volume > 1.5 * 4h volume MA(20);
         Short when close breaks below Donchian lower(20) AND 1d ATR(14) > its 20-period MA AND volume > 1.5 * 4h volume MA(20).
- Exit: ATR-based stoploss (2.0 * ATR(14)) and Donchian middle(20) reversion for profit-taking.
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian channel for structure, volume confirmation for participation,
  1d ATR trend filter to trade only in trending markets (avoiding whipsaws in ranges),
  and ATR for risk management. Designed to work in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (prior 20 periods)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channel from prior 4h OHLC (20-period)
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align 4h Donchian levels to 4h timeframe (no shift needed as we use prior completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_20)
    
    # Get 1d data for ATR(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need 14 for ATR + 20 for MA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])  # first TR is high-low
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period MA of 1d ATR for trend filter
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and its MA to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for 4h stoploss
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h = np.concatenate([[high[0] - low[0]], tr_4h])  # first TR is high-low
    atr14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, Donchian needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_20_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr14_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr_4h = atr14_4h[i]
        
        # Trend filter: 1d ATR(14) > its 20-period MA (trending market)
        trending = atr_1d_aligned[i] > atr_ma_20_aligned[i]
        
        # Volume confirmation: 1.5x threshold (balanced for trade frequency)
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if trending and vol_confirm:
                # Long: Close breaks above Donchian upper (uptrend breakout)
                if curr_close > donchian_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Short: Close breaks below Donchian lower (downtrend breakdown)
                elif curr_close < donchian_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: check exit conditions
            # Stoploss: 2.0 * ATR below entry
            stoploss = entry_price - 2.0 * curr_atr_4h
            # Profit take: close below Donchian middle
            if curr_close < stoploss or curr_close < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            # Stoploss: 2.0 * ATR above entry
            stoploss = entry_price + 2.0 * curr_atr_4h
            # Profit take: close above Donchian middle
            if curr_close > stoploss or curr_close > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Trend_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0