#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian20 + TRIX Momentum + Volume Spike + 1d Trend

HYPOTHESIS: Donchian(20) breakouts capture sustained trends when combined
with TRIX momentum confirmation (catches the start of new trends) and 
volume spike (avoids false breakouts). The 1d SMA(50) keeps us aligned
with macro direction.

WHY IT SHOULD WORK IN BULL AND BEAR:
- Bull (2021, 2024): Breakout above 20-bar high + TRIX turning positive = 
  strong momentum entry. Volume confirms institutional buying.
- Bear (2022): Breakout below 20-bar low + TRIX negative = short rallies
  into broken support. Mean-reversion to broken resistance.
- Range (2023, 2025): False breakouts fail quickly, small losses.
  Whipsaw cost is limited by 2.5x ATR stop.

KEY INSIGHT: Previous failed strategies either:
1. Used too many conflicting indicators (堆砌) → overtrading
2. Had entries that were too strict (0 trades)
3. Ignored volume confirmation (false breakouts)

THIS APPROACH: ONE price structure (Donchian) + ONE momentum (TRIX) + 
volume filter = tight entries = manageable trade count.

TARGET TRADES: 75-150 total over 4 years (19-37/year).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_trix_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(prices, period=9):
    """
    TRIX - Triple EMA Oscillator
    TRIX = 1-period rate of change of triple EMA
    Values near 0 = trend change
    Positive TRIX = bullish momentum
    """
    close = prices if isinstance(prices, pd.Series) else pd.Series(prices)
    
    # Triple EMA
    ema1 = close.ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Rate of change of triple EMA = TRIX
    trix = ema3.pct_change(period=1) * 100  # percentage
    
    return trix.values

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian channel - 20 bar price structure"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = (upper + pd.Series(low).rolling(window=period, min_periods=period).min().values) / 2
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA50 for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === 4h Indicators (all pre-computed before loop) ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian(20) price structure
    dc_upper_20, dc_mid_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # TRIX(9) momentum
    trix_9 = calculate_trix(close, period=9)
    
    # TRIX signal line (4-period SMA of TRIX)
    trix_signal = pd.Series(trix_9).rolling(window=4, min_periods=4).mean().values
    
    # Volume spike detection (1.5x average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 > 1e-10, vol_ma_20, 1.0)
    
    # === Pre-compute session hours (UTC 8-20) ===
    hours = prices.index.hour
    in_session = ((hours >= 8) & (hours <= 20)) | (hours == 0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    
    # Trailing stop tracking
    trailing_stop_long = 0.0
    trailing_stop_short = float('inf')
    
    warmup = 50  # Need at least 20 bars for Donchian + 20 for volume MA
    
    for i in range(warmup, n):
        # === NaN checks ===
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]):
            signals[i] = 0.0
            continue
        if np.isnan(trix_9[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === Update trailing stops if in position ===
        if in_position:
            if position_side > 0:
                # Long: trail stop below highest high since entry
                trailing_stop_long = max(trailing_stop_long, high[i] - 2.0 * entry_atr)
            else:
                # Short: trail stop above lowest low since entry
                trailing_stop_short = min(trailing_stop_short, low[i] + 2.0 * entry_atr)
        
        # === EXIT CONDITIONS ===
        if in_position:
            stop_hit = False
            
            # ATR trailing stop
            if position_side > 0 and low[i] < trailing_stop_long:
                stop_hit = True
            if position_side < 0 and high[i] > trailing_stop_short:
                stop_hit = True
            
            # Opposite 1d trend signal
            if position_side > 0 and close[i] < sma_1d_aligned[i] and (i - entry_bar) >= 4:
                stop_hit = True
            if position_side < 0 and close[i] > sma_1d_aligned[i] and (i - entry_bar) >= 4:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                trailing_stop_long = 0.0
                trailing_stop_short = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # === ENTRY CONDITIONS ===
        # Get previous bar TRIX for crossover detection
        trix_prev = trix_9[i-1] if i > warmup else 0.0
        trix_signal_prev = trix_signal[i-1] if i > warmup else 0.0
        
        # === TREND FILTER: 1d SMA50 alignment ===
        bull_trend = close[i] > sma_1d_aligned[i]
        bear_trend = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN(20) BREAKOUT CONDITIONS ===
        # Long: price breaks above 20-bar high
        bullish_breakout = close[i] > dc_upper_20[i] and close[i-1] <= dc_upper_20[i-1]
        # Short: price breaks below 20-bar low
        bearish_breakout = close[i] < dc_lower_20[i] and close[i-1] >= dc_lower_20[i-1]
        
        # === TRIX MOMENTUM CONFIRMATION ===
        # Long: TRIX crosses above signal line (momentum accelerating)
        trix_bullish = trix_9[i] > trix_signal[i] and trix_prev <= trix_signal_prev
        # Alternative: TRIX already positive and rising (confirm established uptrend)
        trix_momentum_up = trix_9[i] > 0 and trix_9[i] > trix_prev
        # Short: TRIX crosses below signal line
        trix_bearish = trix_9[i] < trix_signal[i] and trix_prev >= trix_signal_prev
        # Alternative: TRIX already negative and falling
        trix_momentum_down = trix_9[i] < 0 and trix_9[i] < trix_prev
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === LONG ENTRY ===
        # Breakout + TRIX momentum + Volume + 1d uptrend
        if bullish_breakout and (trix_bullish or trix_momentum_up) and vol_spike and bull_trend:
            in_position = True
            position_side = 1
            entry_atr = atr_14[i]
            entry_bar = i
            trailing_stop_long = high[i] - 2.0 * entry_atr
            signals[i] = SIZE
        
        # === SHORT ENTRY ===
        # Breakdown + TRIX momentum + Volume + 1d downtrend
        elif bearish_breakout and (trix_bearish or trix_momentum_down) and vol_spike and bear_trend:
            in_position = True
            position_side = -1
            entry_atr = atr_14[i]
            entry_bar = i
            trailing_stop_short = low[i] + 2.0 * entry_atr
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals