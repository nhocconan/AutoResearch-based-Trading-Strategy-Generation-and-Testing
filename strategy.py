#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d trend filter and session filter (08-20 UTC)
# - Long when price breaks above H3 pivot AND 4h EMA(21) > EMA(55) AND 1d close > open (bullish daily candle)
# - Short when price breaks below L3 pivot AND 4h EMA(21) < EMA(55) AND 1d close < open (bearish daily candle)
# - Exit when price returns to Pivot Point (mean reversion to equilibrium)
# - Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity periods
# - Uses discrete position sizing (0.20) to minimize fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots work well in ranging markets; 4h/1d filters ensure alignment with higher timeframe trend
# - Session filter reduces noise trades during Asian session and overnight lows

name = "1h_4h_1d_camarilla_breakout_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 55:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 55 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 4h EMA trend filter: EMA(21) vs EMA(55)
    close_4h = df_4h['close'].values
    ema_21 = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_55 = pd.Series(close_4h).ewm(span=55, min_periods=55, adjust=False).mean().values
    ema_bullish = ema_21 > ema_55
    ema_bearish = ema_21 < ema_55
    
    # Align 4h EMA trend to 1h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_4h, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_4h, ema_bearish)
    
    # Pre-compute 1d daily candle direction: bullish if close > open
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_bullish = close_1d > open_1d
    daily_bearish = close_1d < open_1d
    
    # Align 1d daily direction to 1h timeframe
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Pre-compute Camarilla pivot levels for 1h timeframe
    # Using previous bar's high, low, close for pivot calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Shift by 1 to use previous bar's data (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first value to NaN since we don't have previous bar
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot point (PP)
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Calculate Camarilla levels
    range_hl = prev_high - prev_low
    h3 = pp + (range_hl * 1.1 / 4)
    l3 = pp - (range_hl * 1.1 / 4)
    h4 = pp + (range_hl * 1.1 / 2)
    l4 = pp - (range_hl * 1.1 / 2)
    
    # Breakout conditions: price breaks above H3 or below L3
    breakout_long = close > h3
    breakout_short = close < l3
    
    # Exit condition: price returns to pivot point (PP)
    exit_long = close < pp  # Exit long when price falls below PP
    exit_short = close > pp  # Exit short when price rises above PP
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 since we use previous bar data
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]) or
            np.isnan(breakout_long[i]) or np.isnan(breakout_short[i]) or
            np.isnan(exit_long[i]) or np.isnan(exit_short[i]) or
            np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            not in_session[i]):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 AND 4h bullish trend AND daily bullish candle
            if (breakout_long[i] and 
                ema_bullish_aligned[i] and 
                daily_bullish_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short when price breaks below L3 AND 4h bearish trend AND daily bearish candle
            elif (breakout_short[i] and 
                  ema_bearish_aligned[i] and 
                  daily_bearish_aligned[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion)
            # Exit when price returns to pivot point (PP)
            if position == 1:  # Long position
                if exit_long[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # Short position
                if exit_short[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals