#!/usr/bin/env python3
"""
Experiment #8339: 6-hour Williams Alligator + Elder Ray + 12h trend filter
Hypothesis: In trending markets (12h close above/below EMA50), Williams Alligator 
(jaw/teeth/lips aligned) confirms direction while Elder Ray (bull/bear power) 
filters for strength. Enter on bull/bear power confirmation with Alligator alignment.
Aims for 50-150 total trades over 4 years by requiring multiple confluence factors.
Works in bull (buy strength) and bear (sell weakness) via symmetric long/short logic.
"""

from mtf_data import get_ath_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8339_6w_alligator_elder_12h"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
ELDER_RAY_PERIOD = 13
EMA_TREND_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    trend_up = close_12h > ema_12h   # bullish trend
    trend_down = close_12h < ema_12h  # bearish trend
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator: SMMA (smoothed MA) of median price
    median_price = (high + low) / 2.0
    # Jaw (13), Teeth (8), Lips (5) - all SMMA
    def smma(series, period):
        s = pd.Series(series)
        # First value is SMA, then smoothed
        sma = s.rolling(window=period, min_periods=period).mean()
        result = np.full_like(s, np.nan, dtype=float)
        for i in range(len(s)):
            if i < period:
                result[i] = sma[i]
            else:
                if not np.isnan(sma[i]):
                    result[i] = (result[i-1] * (period-1) + sma[i]) / period
                else:
                    result[i] = result[i-1]
        return result
    
    jaw = smma(median_price, ALLIGATOR_PERIOD)
    teeth = smma(median_price, ALLIGATOR_PERIOD - 3)  # 8
    lips = smma(median_price, ALLIGATOR_PERIOD - 8)  # 5
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD, ELDER_RAY_PERIOD, EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF trend data not available
        if np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Alligator alignment: check if jaws, teeth, lips are aligned in trend direction
        # In uptrend: lips > teeth > jaw
        # In downtrend: lips < teeth < jaw
        if not (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
            alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        else:
            alligator_long = False
            alligator_short = False
        
        # Elder Ray: bull power > 0 and bear power > 0 indicate strength
        bull_strength = bull_power[i] > 0
        bear_strength = bear_power[i] > 0
        
        # Entry conditions require trend + alligator alignment + power confirmation
        long_entry = trend_up_aligned[i] and alligator_long and bull_strength
        short_entry = trend_down_aligned[i] and alligator_short and bear_strength
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals