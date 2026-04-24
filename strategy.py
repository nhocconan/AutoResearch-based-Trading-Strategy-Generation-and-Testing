#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA50 trend filter and 6h volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter and prior OHLC for Camarilla levels.
- Entry: Long when price breaks above Camarilla H3 AND 6h volume > 1.5x 20-period average AND price > 1d EMA50.
         Short when price breaks below Camarilla L3 AND 6h volume > 1.5x 20-period average AND price < 1d EMA50.
- Exit: Opposite Camarilla breakout OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Volume confirmation avoids low-momentum breakouts that often fail.
- 1d EMA50 provides stronger trend filter than shorter EMAs, reducing whipsaws in ranging markets.
- Camarilla H3/L3 levels (from prior 1d OHLC) act as magnet levels where breakouts often continue.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~80 total over 4 years (~20/year) based on strict confluence requirements.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (H3, L3)."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    h3 = pivot + range_hl * 1.1 / 2.0
    l3 = pivot - range_hl * 1.1 / 2.0
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1d OHLC
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    if len(df_1d) >= 2:
        for i in range(1, n):
            prior_day_idx = i - 1
            if prior_day_idx < len(df_1d):
                ph = df_1d['high'].iloc[prior_day_idx]
                pl = df_1d['low'].iloc[prior_day_idx]
                pc = df_1d['close'].iloc[prior_day_idx]
                h3, l3 = camarilla_levels(ph, pl, pc)
                camarilla_h3[i] = h3
                camarilla_l3[i] = l3
    
    # Calculate 6h volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = np.zeros(n)
    volume_spike[:] = np.nan
    for i in range(20, n):
        if not np.isnan(vol_ma_20[i]) and vol_ma_20[i] > 0:
            volume_spike[i] = volume[i] / vol_ma_20[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 20)  # Need 1d EMA50 warmup and 20-period volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN)
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume_ratio = volume_spike[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 1d EMA50
            if position == 1:
                if curr_close < camarilla_l3[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 1d EMA50
            elif position == -1:
                if curr_close > camarilla_h3[i] or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND volume spike > 1.5 AND bullish 1d trend
            if curr_close > camarilla_h3[i] and curr_volume_ratio > 1.5 and curr_close > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND volume spike > 1.5 AND bearish 1d trend
            elif curr_close < camarilla_l3[i] and curr_volume_ratio > 1.5 and curr_close < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA50_TrendFilter_6hVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0