#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter to capture major trend direction.
- Williams %R: Measures overbought/oversold levels on 14-period lookback.
- Entry: Long when Williams %R crosses above -80 from below AND price > 1d EMA50 AND volume > 2.0 * 20-period average volume.
         Short when Williams %R crosses below -20 from above AND price < 1d EMA50 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Williams %R cross OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams %R identifies momentum extremes that often precede reversals, effective in both trending and ranging markets.
- 1d EMA50 provides strong long-term trend filter to avoid counter-trend trades during major moves.
- Volume spike confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Williams %R crossover frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def williams_r(high, low, close, period=14):
    """Calculate Williams %R indicator."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 210:  # Need sufficient data for 1d EMA50
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    # Williams %R (14-period)
    wr = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(wr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = wr[i]
        curr_wr_prev = wr[i-1] if i > 0 else -50
        
        # Exit conditions: opposite Williams %R cross OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: Williams %R crosses below -80 from above OR price falls below 1d EMA50
            if position == 1:
                if curr_wr < -80 and curr_wr_prev >= -80 or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above -20 from below OR price rises above 1d EMA50
            elif position == -1:
                if curr_wr > -20 and curr_wr_prev <= -20 or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R cross with trend filter and volume confirmation
        if position == 0:
            # Williams %R crosses above -80 from below (bullish)
            wr_bullish_cross = curr_wr > -80 and curr_wr_prev <= -80
            # Williams %R crosses below -20 from above (bearish)
            wr_bearish_cross = curr_wr < -20 and curr_wr_prev >= -20
            
            # Long: Bullish Williams %R cross AND price > 1d EMA50 AND volume confirmation
            if wr_bullish_cross and curr_close > ema50_1d_aligned[i] and curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Williams %R cross AND price < 1d EMA50 AND volume confirmation
            elif wr_bearish_cross and curr_close < ema50_1d_aligned[i] and curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0