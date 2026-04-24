#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout + 12h EMA trend + ATR stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA trend filter.
- Entry: Long when price breaks above Donchian(20) high AND price > 12h EMA50.
         Short when price breaks below Donchian(20) low AND price < 12h EMA50.
- Exit: ATR-based stoploss (2.5 * ATR) or opposite Donchian breakout.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels identify volatility breakouts.
- 12h EMA50 provides trend filter to avoid counter-trend trades.
- ATR stoploss manages risk during adverse moves.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period):
    """Calculate Donchian Channels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 4h ATR for stoploss
    atr_4h = atr(high, low, close, 14)
    
    # Donchian channels on 4h (20-period)
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        
        # Check stoploss
        if position != 0:
            stop_loss_hit = False
            if position == 1:  # Long position
                if curr_close <= entry_price - 2.5 * atr_4h[i]:
                    stop_loss_hit = True
            elif position == -1:  # Short position
                if curr_close >= entry_price + 2.5 * atr_4h[i]:
                    stop_loss_hit = True
            
            if stop_loss_hit:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below Donchian low
            if position == 1:
                if curr_close < donch_lo[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
            # Exit short: price breaks above Donchian high
            elif position == -1:
                if curr_close > donch_hi[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter
        if position == 0:
            # Long: price breaks above Donchian high AND bullish 12h trend
            if curr_close > donch_hi[i] and curr_close > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian low AND bearish 12h trend
            elif curr_close < donch_lo[i] and curr_close < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_ATRStop_v1"
timeframe = "4h"
leverage = 1.0