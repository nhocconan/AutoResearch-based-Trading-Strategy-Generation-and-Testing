#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and ATR-based volatility regime filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter to capture major trend direction and avoid counter-trend trades.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 measures bull/bear strength relative to trend.
- Entry: Long when Bull Power > 0 AND rising AND price > 1d EMA50 AND ATR(14) < ATR(50) (low volatility regime).
         Short when Bear Power < 0 AND falling AND price < 1d EMA50 AND ATR(14) < ATR(50).
- Exit: Opposite Elder Ray signal OR price crosses 1d EMA50 in opposite direction OR volatility expands (ATR(14) > 1.5 * ATR(50)).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Elder Ray identifies momentum extremes that often precede reversals, effective in both trending and ranging markets.
- 1d EMA50 provides strong long-term trend filter to avoid counter-trend trades during major moves.
- ATR regime filter ensures trades occur in low volatility environments, reducing whipsaw in choppy markets.
- Estimated trades: ~100 total over 4 years (~25/year) based on Elder Ray crossover frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    return atr_values.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
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
    
    # Calculate Elder Ray components (using 13-period EMA as base)
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate ATR for volatility regime filter
    atr_14 = atr(high, low, close, 14)
    atr_50 = atr(high, low, close, 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 100  # Need sufficient data for ATR50 and EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_bull_power_prev = bull_power[i-1] if i > 0 else 0
        curr_bear_power_prev = bear_power[i-1] if i > 0 else 0
        curr_atr_14 = atr_14[i]
        curr_atr_50 = atr_50[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: Bear Power becomes negative OR price falls below 1d EMA50 OR volatility expands
            if position == 1:
                if (curr_bear_power < 0 and curr_bear_power_prev >= 0) or \
                   curr_close < ema50_1d_aligned[i] or \
                   curr_atr_14 > 1.5 * curr_atr_50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power becomes positive OR price rises above 1d EMA50 OR volatility expands
            elif position == -1:
                if (curr_bull_power > 0 and curr_bull_power_prev <= 0) or \
                   curr_close > ema50_1d_aligned[i] or \
                   curr_atr_14 > 1.5 * curr_atr_50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions
        if position == 0:
            # Low volatility regime filter
            low_vol_regime = curr_atr_14 < curr_atr_50
            
            # Long: Bull Power positive AND rising AND price > 1d EMA50 AND low volatility
            if (curr_bull_power > 0 and curr_bull_power > curr_bull_power_prev and
                curr_close > ema50_1d_aligned[i] and low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND falling AND price < 1d EMA50 AND low volatility
            elif (curr_bear_power < 0 and curr_bear_power < curr_bear_power_prev and
                  curr_close < ema50_1d_aligned[i] and low_vol_regime):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA50_TrendFilter_ATRRegime_v1"
timeframe = "6h"
leverage = 1.0