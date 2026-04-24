#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) mean reversion with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend filter to capture intermediate trend direction.
- RSI(2): Identifies short-term extreme momentum for mean reversion entries.
- Entry: Long when RSI(2) < 10 AND price > 4h EMA50 AND volume > 1.5 * 20-period average volume.
         Short when RSI(2) > 90 AND price < 4h EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: RSI(2) crosses above 50 for longs OR below 50 for shorts.
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- RSI(2) captures short-term exhaustion in both bull and bear markets.
- 4h EMA50 filter ensures trades align with intermediate trend, reducing counter-trend whipsaws.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on RSI(2) extreme frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(values, period):
    """Calculate Relative Strength Index with proper min_periods."""
    delta = pd.Series(values).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 4h volume average for confirmation
    if len(df_4h) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = df_4h['volume'].values / (vol_ma_20 + 1e-10)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # RSI(2) for mean reversion signals
    rsi_2 = rsi(close, 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for 4h EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ratio_4h_aligned[i]) or
            np.isnan(rsi_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_rsi = rsi_2[i]
        curr_rsi_prev = rsi_2[i-1] if i > 0 else 50
        
        # Exit conditions: RSI(2) crosses above 50 for longs OR below 50 for shorts
        if position != 0:
            # Exit long: RSI(2) crosses above 50 from below
            if position == 1:
                if curr_rsi > 50 and curr_rsi_prev <= 50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: RSI(2) crosses below 50 from above
            elif position == -1:
                if curr_rsi < 50 and curr_rsi_prev >= 50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions: RSI(2) extreme with trend filter and volume confirmation
        if position == 0:
            # Long: RSI(2) < 10 (oversold) AND price > 4h EMA50 AND volume confirmation
            long_condition = (curr_rsi < 10 and 
                            curr_close > ema50_4h_aligned[i] and
                            curr_volume > 1.5 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False)
            
            # Short: RSI(2) > 90 (overbought) AND price < 4h EMA50 AND volume confirmation
            short_condition = (curr_rsi > 90 and 
                             curr_close < ema50_4h_aligned[i] and
                             curr_volume > 1.5 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_RSI2_4hEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0