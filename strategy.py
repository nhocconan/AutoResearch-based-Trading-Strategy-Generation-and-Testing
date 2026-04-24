#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation, and ATR-based stoploss.
- Primary timeframe: 4h for balanced trade frequency and signal quality.
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian Channel: 20-period high/low on 4h for breakout signals.
- Volume: Current 4h volume > 1.5 * 20-period volume MA to confirm institutional participation.
- ATR Stoploss: Exit when price moves against position by 2.5 * ATR(20).
- Entry: Long when price breaks above Donchian upper band AND 12h EMA50 bullish AND volume spike.
         Short when price breaks below Donchian lower band AND 12h EMA50 bearish AND volume spike.
- Exit: Opposite Donchian breakout OR ATR stoploss hit OR loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown control.
- Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.
This strategy captures strong momentum moves in the direction of the higher timeframe trend,
filtered by volume confirmation to avoid false breakouts, with ATR stops to manage risk.
Works in both bull and bear markets by only taking trend-aligned breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h Donchian Channel(20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price > Donchian upper AND 12h EMA50 bullish (close > EMA)
                if curr_close > donch_high[i] and curr_close > ema_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price < Donchian lower AND 12h EMA50 bearish (close < EMA)
                elif curr_close < donch_low[i] and curr_close < ema_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            # Opposite Donchian breakout
            if curr_close < donch_low[i]:
                exit_signal = True
            # ATR stoploss: price < entry_price - 2.5 * ATR
            elif curr_close < entry_price - 2.5 * atr[i]:
                exit_signal = True
            # Loss of volume confirmation
            elif not volume_spike[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            # Opposite Donchian breakout
            if curr_close > donch_high[i]:
                exit_signal = True
            # ATR stoploss: price > entry_price + 2.5 * ATR
            elif curr_close > entry_price + 2.5 * atr[i]:
                exit_signal = True
            # Loss of volume confirmation
            elif not volume_spike[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0