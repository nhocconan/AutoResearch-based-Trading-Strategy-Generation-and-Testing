#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d trend filter (EMA50) + volume confirmation + ATR stoploss.
# Donchian breakout provides clear structure, 1d EMA50 filters counter-trend trades,
# volume confirms momentum breakout, ATR stoploss manages risk.
# Works in bull/bear: trend filter avoids counter-trend trades, breakout captures momentum.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get HTF data once before loop
    df_4h = get_htf_data(prices, '4h')  # for ATR calculation
    df_1d = get_htf_data(prices, '1d')  # for trend filter
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: ATR for stoploss and volatility filter ===
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    close_4h = pd.Series(df_4h['close'].values)
    tr1 = high_4h - low_4h
    tr2 = abs(high_4h - close_4h.shift(1))
    tr3 = abs(low_4h - close_4h.shift(1))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === 4h Donchian(20) channels ===
    donch_high_20 = high_4h.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_4h.rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume filter: current 4h volume > 1.5x 20-bar 4h volume SMA ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_sma_20 * 1.5)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian(20) high
        # 2. 1d price above EMA50 (bullish long-term trend bias)
        # 3. Volume confirmation
        if (close[i] > donch_high_20_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian(20) low
        # 2. 1d price below EMA50 (bearish long-term trend bias)
        # 3. Volume confirmation
        elif (close[i] < donch_low_20_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm[i]):
            signals[i] = -0.25
        
        # === STOPLOSS: ATR-based trailing stop ===
        else:
            # Track position from previous bar
            prev_signal = signals[i-1]
            
            if prev_signal > 0:  # Long position
                # Calculate stop level: highest high since entry minus 2*ATR
                # Simplified: use current high - 2*ATR as trailing stop proxy
                if close[i] < (high[i] - 2.0 * atr_4h_aligned[i]):
                    signals[i] = 0.0  # stoploss hit
                else:
                    signals[i] = prev_signal  # maintain position
            elif prev_signal < 0:  # Short position
                # Calculate stop level: lowest low since entry plus 2*ATR
                # Simplified: use current low + 2*ATR as trailing stop proxy
                if close[i] > (low[i] + 2.0 * atr_4h_aligned[i]):
                    signals[i] = 0.0  # stoploss hit
                else:
                    signals[i] = prev_signal  # maintain position
            else:
                signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_EMA50_Vol_ATRStop_v1"
timeframe = "4h"
leverage = 1.0