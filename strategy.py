#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend filter (price above/below weekly EMA50).
- Entry: Long when price breaks above Camarilla R3 AND weekly trend bullish AND volume > 1.5x 20-period average.
         Short when price breaks below Camarilla S3 AND weekly trend bearish AND volume > 1.5x 20-period average.
- Exit: Opposite Camarilla breakout (S3 for longs, R3 for shorts) OR weekly trend reversal.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla pivot levels provide intraday support/resistance based on prior day's range.
- Weekly EMA50 filter ensures trading with the higher-timeframe trend to avoid counter-trend whipsaws.
- Volume confirmation adds conviction to breakouts, reducing false signals.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Camarilla breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate Camarilla pivot levels from prior 12h bar (using prior bar's high/low/close)
    # Camarilla levels: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    # Shift high/low/close by 1 to get prior bar's values
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    # First bar will have invalid prior values (set to 0)
    prior_high[0] = 0
    prior_low[0] = 0
    prior_close[0] = 0
    
    # Calculate Camarilla levels
    camarilla_pp = (prior_high + prior_low + prior_close) / 3
    camarilla_range = prior_high - prior_low
    camarilla_r3 = prior_close + (camarilla_range * 1.1 / 4)
    camarilla_s3 = prior_close - (camarilla_range * 1.1 / 4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * volume_ma)
    
    # Trend conditions
    trend_bullish = close > ema50_1w_aligned
    trend_bearish = close < ema50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for volume MA and prior bar values
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume_confirmed = volume_confirmed[i]
        
        # Exit conditions: opposite Camarilla breakout OR weekly trend reversal
        if position != 0:
            # Exit long: price breaks below S3 OR weekly trend turns bearish
            if position == 1:
                if curr_close < camarilla_s3[i] or not trend_bullish[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R3 OR weekly trend turns bullish
            elif position == -1:
                if curr_close > camarilla_r3[i] or not trend_bearish[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend alignment
        if position == 0:
            # Long: price breaks above R3 AND volume confirmed AND bullish weekly trend
            if curr_close > camarilla_r3[i] and curr_volume_confirmed and trend_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume confirmed AND bearish weekly trend
            elif curr_close < camarilla_s3[i] and curr_volume_confirmed and trend_bearish[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0