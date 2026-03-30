#!/usr/bin/env python3
"""
Experiment #010: 1d TRIX Momentum + Volume + 1w SMA Trend

HYPOTHESIS: TRIX (momentum oscillator) with 1w SMA trend filter should work
across all market conditions:
- 2021 bull: TRIX positive cross + price above 1w SMA = strong momentum continuation
- 2022 bear: TRIX negative cross + price below 1w SMA = momentum continuation short
- 2025 range: TRIX crosses fail near 1w SMA = no trades (correct avoidance)

WHY 1d: Fewer trades = less fee drag. 1w HTF provides macro context.
TRIX was proven in DB: ETHUSDT test Sharpe 1.32 with volume confirmation.

KEY INSIGHT: TRIX triple-smooths noise better than MACD/EMA for daily bars.
Zero-line crossover is a clear, discrete signal (fewer false flips).

TRADE COUNT: 40-80 total over 4 years (10-20/year).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trix_vol_1w_sma_v1"
timeframe = "1d"
leverage = 1.0

def calculate_trix(close, period=14):
    """
    TRIX: Triple-smoothed rate of change.
    Triple EMA of close, then % change over period.
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Rate of change of triple EMA
    trix = np.zeros(n)
    for i in range(period, n):
        if not np.isnan(ema3.values[i]) and not np.isnan(ema3.values[i - period]):
            if abs(ema3.values[i - period]) > 1e-10:
                trix[i] = ((ema3.values[i] / ema3.values[i - period]) - 1.0) * 100
    
    return trix

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

def calculate_sma(values, period):
    """Simple moving average"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w SMA for macro trend (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    sma_1w_26 = calculate_sma(df_1w['close'].values, 26)  # ~1 quarter
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_26)
    
    # === 1d indicators ===
    trix_14 = calculate_trix(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume confirmation: compare to 20d average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_ema = pd.Series(trix_14).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Need TRIX to stabilize
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(trix_14[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND DETECTION (1w SMA) ===
        htf_bullish = close[i] > sma_1w_aligned[i]
        htf_bearish = close[i] < sma_1w_aligned[i]
        
        # === TRIX MOMENTUM SIGNALS ===
        # Positive TRIX = bullish momentum
        # TRIX > 0 and rising (above signal line) = strong bullish
        trix_positive = trix_14[i] > 0
        trix_negative = trix_14[i] < 0
        trix_cross_up = (trix_14[i] > 0) and (trix_14[i-1] <= 0 if i > warmup else False)
        trix_cross_down = (trix_14[i] < 0) and (trix_14[i-1] >= 0 if i > warmup else False)
        
        # TRIX momentum strengthening (above signal line)
        trix_bullish = trix_14[i] > trix_ema[i] if not np.isnan(trix_ema[i]) else False
        trix_bearish = trix_14[i] < trix_ema[i] if not np.isnan(trix_ema[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === MINIMUM HOLD: 3 bars (3 days) to avoid chop ===
        min_hold = (i - entry_bar) >= 3
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on trend reversal (TRIX flips)
            if position_side > 0 and trix_negative and min_hold:
                stop_hit = True
            if position_side < 0 and trix_positive and min_hold:
                stop_hit = True
            
            # Exit on HTF trend failure
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: TRIX crosses positive + volume + 1w uptrend
            if trix_cross_up and vol_spike and htf_bullish:
                in_position = True
                position_side = 1
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG: TRIX already positive + strengthening + 1w uptrend (continuation)
            elif trix_positive and trix_bullish and vol_spike and htf_bullish:
                in_position = True
                position_side = 1
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE * 0.5  # Half for continuation (less conviction)
            
            # SHORT: TRIX crosses negative + volume + 1w downtrend
            elif trix_cross_down and vol_spike and htf_bearish:
                in_position = True
                position_side = -1
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT: TRIX already negative + weakening + 1w downtrend (continuation)
            elif trix_negative and trix_bearish and vol_spike and htf_bearish:
                in_position = True
                position_side = -1
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.5  # Half for continuation
            
            else:
                signals[i] = 0.0
    
    return signals