#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 1d Volume Spike and Chop Regime Filter
Hypothesis: Camarilla H3 (resistance) and L3 (support) levels act as key intraday pivot points.
Breakouts above H3 or below L3 with volume confirmation indicate institutional participation.
Choppiness index regime filter avoids false breakouts in ranging markets (CHOP > 61.8 = range).
In bull markets, we take long breakouts above H3 with uptrend bias; in bear markets,
we take short breakdowns below L3 with downtrend bias. Discrete sizing (0.0, ±0.25) minimizes fee churn.
Target: 20-40 trades/year on 4h.
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR14 for volume spike threshold
    tr1d = np.maximum(
        np.maximum(df_1d['high'].values - df_1d['low'].values,
                   np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))),
        np.abs(np.roll(df_1d['close'].values, 1) - df_1d['low'].values)
    )
    tr1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_14_1d = pd.Series(tr1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h ATR14 for stoploss
    tr = np.maximum(
        np.maximum(high - low,
                   np.abs(high - np.roll(close, 1))),
        np.abs(np.roll(close, 1) - low)
    )
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h Choppiness Index (CHOP) for regime filter
    def calculate_chop(high, low, close, window=14):
        tr = np.maximum(
            np.maximum(high - low,
                       np.abs(high - np.roll(close, 1))),
            np.abs(np.roll(close, 1) - low)
        )
        tr[0] = high[0] - low[0]
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        chop = np.where((hh - ll) > 0, -100 * np.log10(atr_sum / (hh - ll) / np.sqrt(window)), 50)
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ATR and CHOP calculations
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isclose(ema_34_1d_aligned[i], 0) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(chop[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike_threshold = atr_14_1d_aligned[i] * 1.5  # Volume spike: current ATR > 1.5x average ATR
        chop_value = chop[i]
        
        # Calculate Camarilla levels from previous 1d candle
        # We need the completed 1d bar, so use aligned data with additional delay
        # For 1d data aligned to 4h, we use the previous completed 1d bar's OHLC
        idx_1d = i // 96  # Approximate 4h to 1d mapping (96*4h = 1d)
        if idx_1d < 1:
            signals[i] = 0.0
            continue
        
        # Get previous completed 1d bar's OHLC (using align_htf_to_ltf would give current forming bar)
        # Instead, we shift the 1d data by 1 bar to get completed bar
        if idx_1d >= len(df_1d):
            signals[i] = 0.0
            continue
        
        prev_close_1d = df_1d['close'].values[idx_1d - 1]
        prev_high_1d = df_1d['high'].values[idx_1d - 1]
        prev_low_1d = df_1d['low'].values[idx_1d - 1]
        
        # Camarilla levels
        range_1d = prev_high_1d - prev_low_1d
        if range_1d <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_h3 = prev_close_1d + range_1d * 1.1 / 4
        camarilla_l3 = prev_close_1d - range_1d * 1.1 / 4
        camarilla_h4 = prev_close_1d + range_1d * 1.1 / 2
        camarilla_l4 = prev_close_1d - range_1d * 1.1 / 2
        
        # Volume confirmation: current volume > 1.5x average 4h volume
        # Calculate average volume using rolling mean
        if i >= 20:
            avg_volume = np.mean(volume[max(0, i-20):i])
        else:
            avg_volume = np.mean(volume[:i]) if i > 0 else 0
        volume_confirmed = curr_volume > avg_volume * 1.5 if avg_volume > 0 else False
        
        # Regime filter: avoid ranging markets (CHOP > 61.8 = range)
        chop_regime_filter = chop_value < 61.8  # Trending market
        
        if position == 0:
            # Look for entry signals
            # Long: break above H3 with volume confirmation, uptrend, and trending regime
            long_entry = (curr_close > camarilla_h3 and
                         curr_high > camarilla_h3 and  # Ensure breakout candle closes above level
                         volume_confirmed and
                         curr_close > ema_trend and
                         chop_regime_filter)
            # Short: break below L3 with volume confirmation, downtrend, and trending regime
            short_entry = (curr_close < camarilla_l3 and
                          curr_low < camarilla_l3 and  # Ensure breakdown candle closes below level
                          volume_confirmed and
                          curr_close < ema_trend and
                          chop_regime_filter)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: close below L3 (reversal) OR ATR-based stoploss
            exit_signal = (curr_close < camarilla_l3) or (curr_close < entry_price - 2.0 * atr_14[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: close above H3 (reversal) OR ATR-based stoploss
            exit_signal = (curr_close > camarilla_h3) or (curr_close > entry_price + 2.0 * atr_14[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dVolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0