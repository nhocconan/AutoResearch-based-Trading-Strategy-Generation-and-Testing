#!/usr/bin/env python3
"""
Hypothesis: 1d TRIX with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter to capture major trend direction.
- TRIX: Triple-smoothed EMA, reduces noise and identifies momentum shifts.
- Entry: Long when TRIX crosses above zero AND price > 1w EMA50 AND volume > 1.8 * 20-period average volume.
         Short when TRIX crosses below zero AND price < 1w EMA50 AND volume > 1.8 * 20-period average volume.
- Exit: Opposite TRIX cross OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- TRIX is effective in both trending and ranging markets, with less whipsaw than MACD.
- 1w EMA50 provides strong long-term trend filter to avoid counter-trend trades during major moves.
- Volume spike confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~60 total over 4 years (~15/year) based on TRIX zero-cross frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def trix(close, period=15):
    """Calculate TRIX indicator."""
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    trix_raw = 100 * (ema3.pct_change())
    return trix_raw.values

def generate_signals(prices):
    n = len(prices)
    if n < 210:  # Need sufficient data for 1w EMA50
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 1w volume average for confirmation
    if len(df_1w) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = df_1w['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w, additional_delay_bars=1)
    
    # TRIX (15-period)
    trix_val = trix(close, 15)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i]) or
            np.isnan(trix_val[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_trix = trix_val[i]
        curr_trix_prev = trix_val[i-1] if i > 0 else 0.0
        
        # Exit conditions: opposite TRIX cross OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: TRIX crosses below zero from above OR price falls below 1w EMA50
            if position == 1:
                if curr_trix < 0 and curr_trix_prev >= 0 or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: TRIX crosses above zero from below OR price rises above 1w EMA50
            elif position == -1:
                if curr_trix > 0 and curr_trix_prev <= 0 or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: TRIX zero-cross with trend filter and volume confirmation
        if position == 0:
            # TRIX crosses above zero from below (bullish)
            trix_bullish_cross = curr_trix > 0 and curr_trix_prev <= 0
            # TRIX crosses below zero from above (bearish)
            trix_bearish_cross = curr_trix < 0 and curr_trix_prev >= 0
            
            # Long: Bullish TRIX cross AND price > 1w EMA50 AND volume confirmation
            if trix_bullish_cross and curr_close > ema50_1w_aligned[i] and curr_volume > 1.8 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False:
                signals[i] = 0.25
                position = 1
            # Short: Bearish TRIX cross AND price < 1w EMA50 AND volume confirmation
            elif trix_bearish_cross and curr_close < ema50_1w_aligned[i] and curr_volume > 1.8 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_TRIX_1wEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0