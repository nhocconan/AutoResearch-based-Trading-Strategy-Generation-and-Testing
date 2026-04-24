#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter and Williams %R calculation.
- Entry: Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.5 * 4h volume MA(20);
         Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.5 * 4h volume MA(20).
- Exit: Williams %R crosses above -50 for longs or below -50 for shorts, or trend change (price crosses 1d EMA50 opposite direction).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies overextended moves; EMA50 ensures we trade with the daily trend; volume spike confirms institutional participation.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend) with trend filter to avoid counter-trend whipsaws.
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
    
    # Get 1d data for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions
        if position != 0:
            if position == 1:
                # Exit long: Williams %R crosses above -50 OR price breaks below EMA50 (trend change)
                if williams_r_aligned[i] > -50 or curr_close < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            elif position == -1:
                # Exit short: Williams %R crosses below -50 OR price breaks above EMA50 (trend change)
                if williams_r_aligned[i] < -50 or curr_close > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions with volume confirmation and Williams %R extremes
        vol_confirm = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Oversold AND uptrend
                if oversold and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Overbought AND downtrend
                elif overbought and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0