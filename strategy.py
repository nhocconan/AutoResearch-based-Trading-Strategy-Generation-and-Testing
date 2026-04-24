#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1w/1d regime filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for bull/bear regime (price above/below 50w EMA) and 1d for trend filter (price above/below 200d EMA).
- Entry: Long when Elder Ray bull power > 0 AND 1w regime bullish (price > 50w EMA) AND 1d trend bullish (price > 200d EMA).
         Short when Elder Ray bear power < 0 AND 1w regime bearish (price < 50w EMA) AND 1d trend bearish (price < 200d EMA).
- Exit: Opposite Elder Ray signal OR price crosses 200d EMA in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Elder Ray measures bull/bear power behind the move (bull power = high - EMA13, bear power = low - EMA13).
- Weekly regime filter avoids fighting the long-term trend; daily trend filter ensures intermediate alignment.
- Works in bull markets (buy when all bullish aligned) and bear markets (sell when all bearish aligned).
- Estimated trades: ~100 total over 4 years (~25/year) based on Elder Ray signal frequency with dual timeframe filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w regime filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Elder Ray on 6h (bull power = high - EMA13, bear power = low - EMA13)
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Regime and trend conditions
    regime_bullish = close > ema50_1w_aligned  # Above 50w EMA = bull regime
    regime_bearish = close < ema50_1w_aligned  # Below 50w EMA = bear regime
    trend_bullish = close > ema200_1d_aligned  # Above 200d EMA = bull trend
    trend_bearish = close < ema200_1d_aligned  # Below 200d EMA = bear trend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for 200d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Elder Ray signal OR price crosses 200d EMA in opposite direction
        if position != 0:
            # Exit long: Elder Ray turns bearish OR price falls below 200d EMA
            if position == 1:
                if bear_power[i] < 0 or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Elder Ray turns bullish OR price rises above 200d EMA
            elif position == -1:
                if bull_power[i] > 0 or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: All aligned in same direction
        if position == 0:
            # Long: Elder Ray bullish AND bullish regime AND bullish trend
            if bull_power[i] > 0 and regime_bullish[i] and trend_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: Elder Ray bearish AND bearish regime AND bearish trend
            elif bear_power[i] < 0 and regime_bearish[i] and trend_bearish[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1w50EMA_Regime_1d200EMA_Trend_v1"
timeframe = "6h"
leverage = 1.0