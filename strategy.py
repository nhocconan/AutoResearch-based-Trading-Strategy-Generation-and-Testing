#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR-based stoploss.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation (volume > 1.5x 20-period SMA).
- Entry: Long when price breaks above Donchian(20) high AND volume confirmed.
         Short when price breaks below Donchian(20) low AND volume confirmed.
- Exit: ATR-based trailing stop (3x ATR) or opposite Donchian breakout.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian channels provide clear breakout levels in trending markets.
- Volume confirmation reduces false breakouts in choppy/range markets.
- ATR stoploss adapts to volatility and limits drawdown.
- Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
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
    
    # Calculate 12h Donchian(20) channels
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-period SMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20, additional_delay_bars=0)
    volume_confirmed = volume > (1.5 * volume_sma_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(period, 14, 30)  # Donchian(20), ATR(14), volume confirmation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Update trailing stop for existing positions
        if position != 0:
            if position == 1:  # Long position
                # Trail stop up: max of previous stop or (highest high since entry - 3*ATR)
                # Simplified: trail based on ATR from current levels
                new_stop = curr_high - 3.0 * atr[i]
                stop_price = max(stop_price, new_stop)
                # Exit if price hits stop or breaks below Donchian low
                if curr_low <= stop_price or curr_close < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            else:  # Short position
                # Trail stop down: min of previous stop or (lowest low since entry + 3*ATR)
                new_stop = curr_low + 3.0 * atr[i]
                stop_price = min(stop_price, new_stop)
                # Exit if price hits stop or breaks above Donchian high
                if curr_high >= stop_price or curr_close > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirmed
            if curr_close > donchian_high[i] and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                stop_price = curr_low - 3.0 * atr[i]  # Initial stop below entry
            # Short: price breaks below Donchian low AND volume confirmed
            elif curr_close < donchian_low[i] and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                stop_price = curr_high + 3.0 * atr[i]  # Initial stop above entry
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeConfirm_ATRStop_v1"
timeframe = "12h"
leverage = 1.0