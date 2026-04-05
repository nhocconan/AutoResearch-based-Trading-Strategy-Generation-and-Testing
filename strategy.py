#!/usr/bin/env python3
"""
Experiment #8891: 6h Camarilla Pivot Reversal with Volume Spike and 1d Trend Filter.
Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) identify exhaustion points in ranging markets.
In bull/bear regimes, price often reverses at R3/S3 with volume confirmation.
In strong trends, breakouts through R4/S4 continue. Uses 1d EMA filter to avoid counter-trend trades.
Targets 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.
Works in ranging markets (reversals at R3/S3) and trending markets (breakouts at R4/S4).
"""

from mtf_data import get_htf_ata, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8891_6h_camarilla_reversal_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard multiplier for Camarilla levels
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Pivot point
    pivot = (high + low + close) / 3.0
    # Range
    range_val = high - low
    # Camarilla levels
    r4 = pivot + (range_val * 1.1 * 2)
    r3 = pivot + (range_val * 1.1 * 1.5)
    s3 = pivot - (range_val * 1.1 * 1.5)
    s4 = pivot - (range_val * 1.1 * 2)
    return pivot, r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 1d EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, 
                     np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Initialize arrays for Camarilla levels
    pivot = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        if not np.isnan(prev_high[i]) and not np.isnan(prev_low[i]) and not np.isnan(prev_close[i]):
            _, r3[i], r4[i], s3[i], s4[i] = calculate_camarilla(prev_high[i], prev_low[i], prev_close[i])
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
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
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d price below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_SPIKE_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Reversal conditions at R3/S3 (fade extreme moves)
        reversal_long = (close[i] <= s3[i] and 
                         close[i-1] > s3[i-1] and 
                         bull_bias and 
                         volume_confirmed)
        reversal_short = (close[i] >= r3[i] and 
                          close[i-1] < r3[i-1] and 
                          bear_bias and 
                          volume_confirmed)
        
        # Breakout conditions at R4/S4 (continue strong moves)
        breakout_long = (close[i] >= r4[i] and 
                         close[i-1] < r4[i-1] and 
                         bull_bias and 
                         volume_confirmed)
        breakout_short = (close[i] <= s4[i] and 
                          close[i-1] > s4[i-1] and 
                          bear_bias and 
                          volume_confirmed)
        
        # Entry conditions
        long_entry = reversal_long or breakout_long
        short_entry = reversal_short or breakout_short
        
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
</code> 
</pre>
</body>
</html> 
</html>