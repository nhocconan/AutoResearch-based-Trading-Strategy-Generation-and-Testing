#!/usr/bin/env python3
"""
Experiment #8065: 12-hour Camarilla pivot with 1-day trend filter and volume confirmation.
Hypothesis: Price approaching Camarilla pivot levels (L3/H3) on 12h with volume >1.3x 20-period MA 
and aligned 1d trend (price above/below 1d EMA50) captures reversals in ranging markets and 
breakouts in trending markets. The 1d timeframe provides market regime context to reduce 
whipsaw while maintaining trade frequency of ~12-37/year for 12h timeframe.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8065_12h_camarilla_pivot_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Use previous day's OHLC for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close
    c = close + (range_val * 1.1 / 6)
    l3 = close - (range_val * 1.1 / 4)
    h3 = close + (range_val * 1.1 / 4)
    return l3, c, h3, c  # L3, C, H3, (dummy for symmetry)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Price relative to EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = df_1d['close'].shift(1).values  # Previous day's close
    
    # Calculate L3 and H3 levels for each 1d bar
    l3_levels = np.full_like(close_1d, np.nan)
    h3_levels = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        if not np.isnan(high_1d[i-1]) and not np.isnan(low_1d[i-1]) and not np.isnan(close_1d_shift[i-1]):
            l3, _, h3, _ = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d_shift[i-1])
            l3_levels[i] = l3
            h3_levels[i] = h3
    
    # Align Camarilla levels to 12h timeframe
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_levels)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h3_aligned[i]):
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
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Proximity to Camarilla levels (within 0.5*ATR)
        near_l3 = abs(close[i] - l3_aligned[i]) <= (0.5 * atr[i]) if not np.isnan(l3_aligned[i]) else False
        near_h3 = abs(close[i] - h3_aligned[i]) <= (0.5 * atr[i]) if not np.isnan(h3_aligned[i]) else False
        
        # Entry conditions: fade extreme levels in ranging markets, break in trending
        # In ranging (price near EMA): fade L3/H3
        # In trending (price away from EMA): break L3/H3
        price_vs_ema_level = close[i] - ema_1d[i] if not np.isnan(ema_1d[i]) else 0
        ema_distance = abs(price_vs_ema_level)
        
        # Fade condition: price near L3/H3 and close to EMA (ranging market)
        fade_long = near_l3 and bull_bias and volume_confirmed and (ema_distance < atr[i])
        fade_short = near_h3 and bear_bias and volume_confirmed and (ema_distance < atr[i])
        
        # Break condition: price breaks L3/H3 with strong volume (trending market)
        break_long = (close[i] > h3_aligned[i]) and bull_bias and volume_confirmed
        break_short = (close[i] < l3_aligned[i]) and bear_bias and volume_confirmed
        
        # Generate signals
        if position == 0:
            if fade_long or break_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short or break_short:
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