#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On 4h timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filtering. Enter long when KAMA slopes up, RSI>50, and CHOP<38.2 (trending regime); short when KAMA slopes down, RSI<50, and CHOP<38.2. Avoid ranging markets (CHOP>61.8). ATR-based stoploss limits drawdown. Designed to work in both bull and bear markets via adaptive trend filter and regime avoidance.
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
    
    # Get 1d data for HTF indicators (more stable than 4h for longer-term filters)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA(10,2,30) on 1d close
    close_1d = df_1d['close'].values
    er = np.abs(np.diff(close_1d, 10)) / (
        np.sum(np.abs(np.diff(close_1d, 1)), axis=0)[:len(close_1d)-10] + 1e-10
    )
    # Pad ER array to match close_1d length
    er_full = np.concatenate([np.full(10, np.nan), er])
    sc = (er_full * 0.28 + 0.06) ** 2
    sc = np.where(np.isnan(sc), 0.0, sc)
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_full = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate Choppiness Index(14) on 1d OHLC
    atr_1d = []
    for i in range(1, len(df_1d)):
        tr = max(
            df_1d['high'].values[i] - df_1d['low'].values[i],
            abs(df_1d['high'].values[i] - df_1d['close'].values[i-1]),
            abs(df_1d['low'].values[i] - df_1d['close'].values[i-1])
        )
        atr_1d.append(tr)
    atr_1d = np.concatenate([[np.nan], atr_1d])
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    high_low_range = pd.Series(df_1d['high'].values - df_1d['low'].values).rolling(window=14, min_periods=14).max().values - \
                     pd.Series(df_1d['high'].values - df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / high_low_range) / np.log10(14)
    chop_full = np.concatenate([np.full(13, np.nan), chop])  # 13 NaNs for 14-period indicator
    
    # Align HTF indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_full)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_full)
    
    # ATR(14) for 4h stoploss
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high).rolling(window=2).max() - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).rolling(window=2).min() - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Warmup: need KAMA(10,2,30), RSI(14), CHOP(14), ATR(14)
    start_idx = max(30, 14, 14, 14) + 5  # extra buffer for alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama_aligned[i]
        kama_prev = kama_aligned[i-1] if i > 0 else kama_val
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        
        # KAMA slope: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        # Regime filter: trending market (CHOP < 38.2), avoid ranging (CHOP > 61.8)
        trending_regime = chop_val < 38.2
        ranging_regime = chop_val > 61.8
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, trending regime
            long_signal = kama_rising and (rsi_val > 50) and trending_regime
            
            # Short: KAMA falling, RSI < 50, trending regime
            short_signal = kama_falling and (rsi_val < 50) and trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                atr_at_entry = atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                atr_at_entry = atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ATR-based stoploss or regime change to ranging or KAMA turns down
            if (close_val < entry_price - 2.5 * atr_at_entry or 
                ranging_regime or 
                not kama_rising):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR-based stoploss or regime change to ranging or KAMA turns up
            if (close_val > entry_price + 2.5 * atr_at_entry or 
                ranging_regime or 
                not kama_falling):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0