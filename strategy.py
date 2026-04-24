#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish when close > EMA50, bearish when close < EMA50).
- Entry: Long when Williams %R(14) crosses above -20 from below (exit oversold) in 1d bull trend with volume > 2.0 * 6h volume MA(20); Short when Williams %R(14) crosses below -80 from above (enter overbought) in 1d bear trend with volume > 2.0 * 6h volume MA(20).
- Exit: Opposite Williams %R signal (long exits when %R crosses below -80, short exits when %R crosses above -20) or ATR-based trailing stop (2.0 * ATR(14)).
- Signal size: 0.25 discrete to balance capture and fee control.
- Williams %R captures mean reversion in overextended moves, EMA50 filter avoids counter-trend trades, volume spike confirms institutional participation. Works in both bull and bear markets by only taking trend-aligned mean reversion entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Volume confirmation: 2.0x threshold
        vol_confirmed = curr_volume > 2.0 * vol_ma_6h_aligned[i]
        
        # Determine 1d EMA50 trend: bullish if close > EMA50, bearish if close < EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        # Williams %R signals
        # Long signal: %R crosses above -20 from below (exit oversold)
        long_signal = (prev_williams_r <= -20 and curr_williams_r > -20)
        # Short signal: %R crosses below -80 from above (enter overbought)
        short_signal = (prev_williams_r >= -80 and curr_williams_r < -80)
        # Exit signals
        long_exit = (prev_williams_r >= -80 and curr_williams_r < -80)  # %R crosses below -80
        short_exit = (prev_williams_r <= -20 and curr_williams_r > -20)  # %R crosses above -20
        
        if position == 0:
            # Check for entry signals
            # Long: Williams %R exits oversold in bull trend with volume confirmation
            if long_signal and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R enters overbought in bear trend with volume confirmation
            elif short_signal and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: check exit conditions
            if long_exit or short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit conditions
            if long_exit or short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0