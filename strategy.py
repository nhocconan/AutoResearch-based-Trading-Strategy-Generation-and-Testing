#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA34 trend filter, 1d for session filter.
- Entry: Long when price breaks above Camarilla H3 AND 4h EMA34 > 4h EMA34(previous) (uptrend) AND volume > 1.5 * 1h volume MA(20);
         Short when price breaks below Camarilla L3 AND 4h EMA34 < 4h EMA34(previous) (downtrend) AND volume > 1.5 * 1h volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when EMA34 slope changes sign).
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla H3/L3 breakout frequency with filters.
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
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34_4h - np.roll(ema_34_4h, 1)
    ema_34_slope[0] = 0
    
    # Get 1h data for Camarilla levels (using 1h high/low/close)
    # Camarilla levels based on previous day's OHLC
    # For simplicity, we'll use rolling window of 24 periods (1 day in 1h)
    if len(high) < 24 or len(low) < 24 or len(close) < 24:
        return np.zeros(n)
    
    # Calculate previous day's OHLC (24-period rolling)
    prev_high = pd.Series(high).rolling(window=24, min_periods=24).max().shift(1).values
    prev_low = pd.Series(low).rolling(window=24, min_periods=24).min().shift(1).values
    prev_close = pd.Series(close).rolling(window=24, min_periods=24).last().shift(1).values
    
    # Camarilla levels
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Get 1h data for volume MA
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_34_slope)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)  # Use 4h alignment for daily levels
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    vol_ma_1h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_1h)  # Approximate alignment
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit: trend change (EMA34 slope changes sign)
        if position != 0:
            if position == 1 and ema_34_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_34_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Camarilla breakout
        bullish_breakout = curr_high > camarilla_h3_aligned[i]  # Break above H3
        bearish_breakout = curr_low < camarilla_l3_aligned[i]    # Break below L3
        
        # Trend filter: only trade in direction of 4h EMA34 slope
        uptrend = ema_34_slope_aligned[i] > 0
        downtrend = ema_34_slope_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_1h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Camarilla H3 AND uptrend
                if bullish_breakout and uptrend:
                    signals[i] = 0.20
                    position = 1
                # Short: Price breaks below Camarilla L3 AND downtrend
                elif bearish_breakout and downtrend:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_EMA34_Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0