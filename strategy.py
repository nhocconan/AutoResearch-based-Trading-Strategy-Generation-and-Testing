#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and volume context.
- Entry: Long when Williams %R(14) < -80 (oversold) AND price > 6h EMA50 AND 1d EMA34 > 1d EMA34(previous) (uptrend) AND volume > 2.0 * 6h volume MA(50);
         Short when Williams %R(14) > -20 (overbought) AND price < 6h EMA50 AND 1d EMA34 < 1d EMA34(previous) (downtrend) AND volume > 2.0 * 6h volume MA(50).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when 1d EMA34 slope changes sign against position).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies extreme momentum exhaustion; EMA34 trend filter ensures we trade with the daily trend; volume spike (2.0x) confirms institutional participation.
- Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend) with trend filter to avoid counter-trend whipsaws.
- Estimated trades: ~80 total over 4 years (~20/year) based on Williams %R extreme frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34_1d - np.roll(ema_34_1d, 1)
    ema_34_slope[0] = 0
    
    # Calculate Williams %R(14) on primary 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h EMA50 for dynamic support/resistance
    ema_50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 2.0x 50-period MA
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to primary 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    vol_ma_50_aligned = align_htf_to_ltf(prices, prices, vol_ma_50)  # same timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Williams %R(14) and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(ema_50_6h[i]) or np.isnan(vol_ma_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit: trend change (1d EMA34 slope changes sign against position)
        if position != 0:
            if position == 1 and ema_34_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_34_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Williams %R extremes
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        above_ema50 = curr_close > ema_50_6h[i]
        below_ema50 = curr_close < ema_50_6h[i]
        
        # Trend filter: only trade in direction of 1d EMA34 slope
        uptrend = ema_34_slope_aligned[i] > 0
        downtrend = ema_34_slope_aligned[i] < 0
        
        # Volume confirmation (2.0x average volume)
        vol_confirm = curr_volume > 2.0 * vol_ma_50_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R oversold AND price above EMA50 AND uptrend
                if oversold and above_ema50 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R overbought AND price below EMA50 AND downtrend
                elif overbought and below_ema50 and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_Reversal_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0