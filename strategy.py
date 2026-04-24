#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA trend direction.
- Williams %R(14): identifies overbought/oversold conditions for mean reversion.
- Entry: Long when Williams %R crosses above -80 (from below) AND price > 12h EMA50 AND volume > 1.5 * 20-period average volume.
         Short when Williams %R crosses below -20 (from above) AND price < 12h EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Williams %R crosses above -20 for longs OR below -80 for shorts (mean reversion completion).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R captures reversals at extremes, effective in both bull and bear markets.
- Volume confirmation ensures legitimacy of reversals.
- 12h EMA50 filter ensures trading with the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period):
    """Calculate Williams %R with proper min_periods."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams %R(14)
    wr = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 50 for EMA, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(wr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        prev_wr = wr[i-1]
        curr_wr = wr[i]
        
        # Exit conditions: Williams %R mean reversion completion
        if position != 0:
            # Exit long: Williams %R crosses above -20 (overbought)
            if position == 1:
                if prev_wr < -20 and curr_wr >= -20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses below -80 (oversold)
            elif position == -1:
                if prev_wr > -80 and curr_wr <= -80:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R reversal with trend and volume confirmation
        if position == 0:
            # Williams %R reversal signals
            wr_cross_up = prev_wr < -80 and curr_wr >= -80   # Cross above -80 (oversold)
            wr_cross_down = prev_wr > -20 and curr_wr <= -20 # Cross below -20 (overbought)
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20[i]
            
            # Trend filter: price relative to 12h EMA50
            price_above_ema = curr_close > ema_50_12h_aligned[i]
            price_below_ema = curr_close < ema_50_12h_aligned[i]
            
            if wr_cross_up and volume_confirm and price_above_ema:
                signals[i] = 0.25
                position = 1
            elif wr_cross_down and volume_confirm and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR14_Reversal_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0