#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) mean reversion with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend filter and volume MA.
- Entry: Long when RSI(2) < 10 AND price > 4h EMA50 (uptrend) AND volume > 1.5 * 4h volume MA(20);
         Short when RSI(2) > 90 AND price < 4h EMA50 (downtrend) AND volume > 1.5 * 4h volume MA(20).
- Exit: RSI(2) > 50 for long exit, RSI(2) < 50 for short exit, or trend change (signal=0 when 4h EMA50 slope changes sign).
- Signal size: 0.20 discrete to minimize fee drag.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~100 total over 4 years (~25/year) based on RSI(2) extremes with filters.
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
    
    # Calculate RSI(2) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for EMA50 trend filter and volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA50 and its slope
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = ema_50_4h - np.roll(ema_50_4h, 1)
    ema_50_slope[0] = 0
    
    # Calculate 4h volume MA(20)
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_50_slope)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for RSI(2) and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_50_slope_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions
        if position != 0:
            # Exit on RSI mean reversion or trend change
            if position == 1 and (rsi[i] > 50 or ema_50_slope_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and (rsi[i] < 50 or ema_50_slope_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and trend filter
        bullish_setup = rsi[i] < 10  # Oversold
        bearish_setup = rsi[i] > 90  # Overbought
        
        # Trend filter: only trade in direction of 4h EMA50
        uptrend = ema_50_4h_aligned[i] > ema_50_4h_aligned[max(0, i-1)]  # Rising EMA50
        downtrend = ema_50_4h_aligned[i] < ema_50_4h_aligned[max(0, i-1)]  # Falling EMA50
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: RSI oversold AND uptrend
                if bullish_setup and uptrend:
                    signals[i] = 0.20
                    position = 1
                # Short: RSI overbought AND downtrend
                elif bearish_setup and downtrend:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_RSI2_EMA50_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0