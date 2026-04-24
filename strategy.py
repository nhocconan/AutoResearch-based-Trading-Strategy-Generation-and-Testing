#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend filter to capture intermediate trend direction.
- Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close) measures buying/selling pressure.
- Entry: Long when Bull Power > 0 AND Bull Power rising (current > previous) AND price > 12h EMA50 AND volume > 1.5 * 20-period average volume.
         Short when Bear Power < 0 AND Bear Power falling (current < previous) AND price < 12h EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Elder Ray condition OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Elder Ray identifies the underlying power behind price moves, effective in both trending and ranging markets.
- 12h EMA50 provides intermediate trend filter to avoid counter-trend trades.
- Volume confirmation ensures moves have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Elder Ray crossover frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for 12h EMA50 and Elder Ray
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 12h volume average for confirmation
    if len(df_12h) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h, additional_delay_bars=1)
    
    # Elder Ray (13-period EMA of close)
    ema13_close = ema(close, 13)
    bull_power = high - ema13_close  # Bull Power = High - EMA13(Close)
    bear_power = low - ema13_close   # Bear Power = Low - EMA13(Close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for 12h EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema13_close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_bull_power_prev = bull_power[i-1] if i > 0 else 0
        curr_bear_power_prev = bear_power[i-1] if i > 0 else 0
        curr_ema13 = ema13_close[i]
        
        # Exit conditions: opposite Elder Ray condition OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: Bull Power <= 0 OR price falls below 12h EMA50
            if position == 1:
                if curr_bull_power <= 0 or curr_close < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bear Power >= 0 OR price rises above 12h EMA50
            elif position == -1:
                if curr_bear_power >= 0 or curr_close > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray with trend filter and volume confirmation
        if position == 0:
            # Long: Bull Power > 0 AND Bull Power rising AND price > 12h EMA50 AND volume confirmation
            long_condition = (curr_bull_power > 0 and 
                            curr_bull_power > curr_bull_power_prev and 
                            curr_close > ema50_12h_aligned[i] and
                            curr_volume > 1.5 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False)
            
            # Short: Bear Power < 0 AND Bear Power falling AND price < 12h EMA50 AND volume confirmation
            short_condition = (curr_bear_power < 0 and 
                             curr_bear_power < curr_bear_power_prev and 
                             curr_close < ema50_12h_aligned[i] and
                             curr_volume > 1.5 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_TrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0