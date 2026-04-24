#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and 1d volume spike.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA34 trend filter, 1d for volume confirmation.
- Entry: Long when price breaks above H3 (bullish breakout) AND price > 4h EMA34 AND 1d volume > 1.5 * 20-period average volume.
         Short when price breaks below L3 (bearish breakout) AND price < 4h EMA34 AND 1d volume > 1.5 * 20-period average volume.
- Exit: Opposite Camarilla breakout (price crosses L3 for longs, H3 for shorts) OR 4h EMA34 cross in opposite direction.
- Signal size: 0.20 discrete to minimize fee drag.
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend).
- Camarilla levels provide intraday support/resistance; volume confirms institutional interest.
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h trend filter: EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    ema34_4h = ema(df_4h['close'].values, 34)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h, additional_delay_bars=1)
    
    # Calculate 1d volume average for spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20, additional_delay_bars=1)
    
    # Calculate 1h Camarilla levels (based on previous day's OHLC)
    # We need to resample to daily to get previous day's OHLC, but use actual 1d data
    # For each 1h bar, Camarilla levels are based on the prior 1d bar's OHLC
    # Since we have df_1d with actual daily data, we align it and use prior values
    
    # Get prior day's OHLC (shifted by 1 to avoid look-ahead)
    if len(df_1d) >= 2:
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
    else:
        return np.zeros(n)
    
    # Align prior day's OHLC to 1h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high, additional_delay_bars=1)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low, additional_delay_bars=1)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close, additional_delay_bars=1)
    
    # Calculate Camarilla levels for 1h timeframe
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    rang = prev_high_aligned - prev_low_aligned
    h3 = prev_close_aligned + rang * 1.1 / 4
    l3 = prev_close_aligned - rang * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume spike condition: current 1h volume > 1.5 * 20-period average 1d volume
        # Note: Comparing 1h volume to 1d average volume - this is intentional for volatility filter
        vol_spike = curr_volume > 1.5 * vol_ma_20_aligned[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price crosses below L3 OR price falls below 4h EMA34
            if position == 1:
                if curr_close < l3[i] or curr_close < ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above H3 OR price rises above 4h EMA34
            elif position == -1:
                if curr_close > h3[i] or curr_close > ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume confirmation
        if position == 0:
            # Long: price breaks above H3 AND bullish trend AND volume spike
            if curr_close > h3[i] and curr_close > ema34_4h_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below L3 AND bearish trend AND volume spike
            elif curr_close < l3[i] and curr_close < ema34_4h_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0