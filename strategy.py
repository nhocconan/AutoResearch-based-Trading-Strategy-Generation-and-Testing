#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R reversal with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA34 trend filter.
- Entry: Long when Williams %R(14) crosses above -80 (oversold) AND 1w EMA34 rising AND volume > 1.5 * 1d volume MA(20);
         Short when Williams %R(14) crosses below -20 (overbought) AND 1w EMA34 falling AND volume > 1.5 * 1d volume MA(20).
- Exit: Close-based reversal (opposite signal) or trend change (signal=0 when 1w EMA34 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R captures mean reversion in bear markets; 1w EMA34 trend filter ensures we trade with the weekly trend to avoid counter-trend whipsaws; volume confirmation avoids false signals.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend) with trend filter.
- Estimated trades: ~50 total over 4 years (~12/year) based on Williams %R reversal frequency with filters.
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34_1w - np.roll(ema_34_1w, 1)
    ema_34_slope[0] = 0
    
    # Calculate 1d volume MA(20) for confirmation
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams %R(14) on 1d timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate Williams %R crossovers
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = williams_r[0]  # Initialize first value
    williams_r_cross_above_80 = (williams_r_prev <= -80) & (williams_r > -80)  # Cross above -80 (oversold)
    williams_r_cross_below_20 = (williams_r_prev >= -20) & (williams_r < -20)  # Cross below -20 (overbought)
    
    # Align all indicators to primary 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_34_slope)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, prices, vol_ma_1d)
    williams_r_aligned = align_htf_to_ltf(prices, prices, williams_r)
    williams_r_cross_above_80_aligned = align_htf_to_ltf(prices, prices, williams_r_cross_above_80.astype(float))
    williams_r_cross_below_20_aligned = align_htf_to_ltf(prices, prices, williams_r_cross_below_20.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 34  # Need sufficient data for EMA34 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or
            np.isnan(williams_r_cross_above_80_aligned[i]) or np.isnan(williams_r_cross_below_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit: trend change (1w EMA34 slope changes sign)
        if position != 0:
            if position == 1 and ema_34_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_34_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Williams %R reversal
        vol_confirm = curr_volume > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R crosses above -80 (oversold) AND weekly uptrend
                if williams_r_cross_above_80_aligned[i] > 0.5 and ema_34_slope_aligned[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 (overbought) AND weekly downtrend
                elif williams_r_cross_below_20_aligned[i] > 0.5 and ema_34_slope_aligned[i] < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Reversal_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0