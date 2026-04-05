#!/usr/bin/env python3
"""
Experiment #8379: 6-hour Williams Alligator with 12-hour trend filter and volume confirmation.
Hypothesis: Williams Alligator (SMMA-based system) identifies trend presence and direction on 6h.
Entries occur when price aligns with Alligator jaws/teeth/lips in bullish/bearish configuration,
filtered by 12h EMA trend and volume confirmation. Exits via ATR-based trailing stop.
Designed for 6h timeframe with 50-150 total trades over 4 years.
"""

from mtf_data import get_alt_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8379_6h_williams_alligator_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
ALLIGATOR_Jaws_SHIFT = 8
ALLIGATOR_TEETH_SHIFT = 5
ALLIGATOR_LIPS_SHIFT = 3
EMA_TREND_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_TRAIL_MULTIPLIER = 2.5

def smma(series, period):
    """Smoothed Moving Average (SMMA) as used in Williams Alligator"""
    sma = pd.Series(series).rolling(window=period, min_periods=period).mean()
    # First value is SMA, then recursive smoothing
    smma_vals = np.full_like(series, np.nan, dtype=float)
    if len(series) >= period:
        smma_vals[period-1] = sma.iloc[period-1]
        for i in range(period, len(series)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
    return smma_vals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 12h EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_12h > ema_12h, 1,
                     np.where(close_12h < ema_12h, -1, 0))
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_12h, price_vs_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (all SMMA)
    jaws = smma(high + low, ALLIGATOR_PERIOD)  # Median price
    jaws = np.roll(jaws, ALLIGATOR_Jaws_SHIFT)  # Shift forward
    
    teeth = smma(high + low, ALLIGATOR_PERIOD)
    teeth = np.roll(teeth, ALLIGATOR_TEETH_SHIFT)
    
    lips = smma(high + low, ALLIGATOR_PERIOD)
    lips = np.roll(lips, ALLIGATOR_LIPS_SHIFT)
    
    # Alligator alignment signals
    # Bullish: Lips > Teeth > Jaws (all aligned upward)
    bullish_alignment = (lips > teeth) & (teeth > jaws)
    # Bearish: Lips < Teeth < Jaws (all aligned downward)
    bearish_alignment = (lips < teeth) & (teeth < jaws)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
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
    start = max(ALLIGATOR_PERIOD + ALLIGATOR_Jaws_SHIFT, 
                EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss (trailing stop)
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
            # Update trailing stop
            new_stop = close[i] - (ATR_TRAIL_MULTIPLIER * atr[i])
            if new_stop > stop_price:
                stop_price = new_stop
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
            # Update trailing stop
            new_stop = close[i] + (ATR_TRAIL_MULTIPLIER * atr[i])
            if new_stop < stop_price:
                stop_price = new_stop
        
        # Determine market bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1
        bear_bias = price_vs_ema_aligned[i] == -1
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and bullish_alignment[i] and volume_confirmed
        short_entry = bear_bias and bearish_alignment[i] and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_TRAIL_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_TRAIL_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals