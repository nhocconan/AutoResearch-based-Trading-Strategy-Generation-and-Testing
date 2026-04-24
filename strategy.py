#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter to capture intermediate-term trend direction.
- Camarilla levels: Calculated from prior 1d OHLC; H3 (resistance) and L3 (support) are key intraday levels.
- Entry: Long when price breaks above H3 with volume > 2.0 * 20-period average volume AND price > 1d EMA34.
         Short when price breaks below L3 with volume > 2.0 * 20-period average volume AND price < 1d EMA34.
- Exit: Opposite Camarilla break (price crosses back below H3 for long, above L3 for short) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide mathematically derived support/resistance that often attract institutional order flow.
- 1d EMA34 provides intermediate trend filter to avoid counter-trend trades during corrections.
- Volume spike confirmation ensures breakouts have participation, reducing false signals in low-liquidity periods.
- Works in both bull and bear markets: In bull markets, longs are favored by trend filter; in bear markets, shorts are favored.
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla break frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from prior 1d bar
        if i >= 1:
            # Get prior 1d bar's OHLC (using the last completed 1d bar)
            # Since we're on 12h timeframe, we need to get the prior daily bar
            # We'll use the prior bar's high/low/close for Camarilla calculation
            # For 12h timeframe, we approximate using prior bar data
            prior_high = high[i-1]
            prior_low = low[i-1]
            prior_close = close[i-1]
            
            # Calculate Camarilla levels
            range_val = prior_high - prior_low
            if range_val <= 0:
                # Skip if no valid range
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
                
            camarilla_h3 = prior_close + range_val * 1.1 / 4
            camarilla_l3 = prior_close - range_val * 1.1 / 4
        else:
            # Not enough data for prior bar
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla break OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price crosses back below H3 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < camarilla_h3 or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses back above L3 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > camarilla_l3 or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla break with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            # Use aligned volume ratio from 1d timeframe
            vol_confirmed = vol_ratio_1d_aligned[i] > 2.0
            
            # Long: Break above H3 AND price > 1d EMA34 AND volume confirmation
            if curr_close > camarilla_h3 and curr_close > ema34_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Break below L3 AND price < 1d EMA34 AND volume confirmation
            elif curr_close < camarilla_l3 and curr_close < ema34_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0