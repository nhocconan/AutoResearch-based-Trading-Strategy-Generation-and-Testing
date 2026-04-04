#!/usr/bin/env python3
"""
exp_6693_4h_donchian20_12h_hma_v1
Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
In bull markets: buy breakouts above upper channel when 12h HMA is rising and volume confirms.
In bear markets: sell breakdowns below lower channel when 12h HMA is falling and volume confirms.
Uses discrete position sizing (0.25) and ATR-based stoploss to control drawdown.
Designed for 4h timeframe to target 75-200 trades over 4 years (19-50/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6693_4h_donchian20_12h_hma_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for HMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA(21) for trend
    close_12h = df_12h['close'].values
    hma_12h = calculate_hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
                
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # HMA trend: rising if current > previous, falling if current < previous
        hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 else False
        
        # Donchian breakout/breakdown
        breakout_up = close[i] > highest_high[i-1] if i > 0 else False  # Use previous bar's channel
        breakdown_down = close[i] < lowest_low[i-1] if i > 0 else False
        
        # Enter new positions only if flat
        if position == 0:
            # Long: breakout above upper channel with rising 12h HMA and volume
            if breakout_up and hma_rising and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            # Short: breakdown below lower channel with falling 12h HMA and volume
            elif breakdown_down and hma_falling and vol_confirmed:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(values).rolling(window=half_period, min_periods=half_period).mean().values
    # WMA of full period
    wma_full = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw HMA with sqrt period
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
    return hma