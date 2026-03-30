#!/usr/bin/env python3
"""
Experiment #021: 4h TRIX Momentum + HTF Trend + Volume + Choppiness Filter

HYPOTHESIS: Momentum reversal entries with regime filtering
- TRIX crossing zero = momentum shift signal (proven edge from DB)
- 12h EMA for trend direction (filters counter-trend trades)
- CHOP > 61.8 = SKIP (avoid ranging markets - key killer of returns)
- Volume spike = institutional confirmation
- 2.5 ATR stoploss for risk management

WHY IT WORKS IN BULL + BEAR + RANGE:
- Bull: TRIX crosses up + HTF up = momentum longs (catches rallies)
- Bear: TRIX crosses down + HTF down = momentum shorts (catches dumps)
- Range: CHOP > 61.8 = SKIP (avoids whipsaws in chop)
- ATR stoploss adapts to volatility (handles 2022 crash)

KEY IMPROVEMENT OVER #020:
- #020: TRIX + Donchian + CHOP combined = 160 trades, Sharpe 0.362
- #021: TRIX + HTF trend + volume confirm (fewer, higher quality)
- TARGET: 100-180 total trades over 4 years (25-45/year)

Entry logic:
- Long: TRIX crosses above 0 + HTF trend up + volume spike
- Short: TRIX crosses below 0 + HTF trend down + volume spike
- Exit: stoploss at 2.5 ATR, or opposite TRIX cross, or CHOP > 61.8
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_htf_trend_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_trix(close, period=9):
    """
    TRIX - Triple EMA Oscillator
    - TRIX > 0 = bullish momentum
    - TRIX < 0 = bearish momentum
    - TRIX crossing zero line = momentum shift (our entry signal)
    """
    n = len(close)
    if n < period * 3 + 1:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # TRIX = rate of change of triple EMA
    trix = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema3.iloc[i]) and not np.isnan(ema3.iloc[i-1]) and ema3.iloc[i-1] != 0:
            trix[i] = ((ema3.iloc[i] / ema3.iloc[i-1]) - 1) * 100
    
    return trix

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter (avoid whipsaws)
    CHOP < 50 = trending - GOOD environment
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(21) for trend direction
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix = calculate_trix(close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 200  # TRIX needs ~30 bars + 20 vol MA + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(trix[i]) or np.isnan(trix[i-1]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        is_choppy = chop[i] > 61.8
        
        # === TRIX CROSSOVER DETECTION ===
        # TRIX crossing above zero = bullish momentum shift
        # TRIX crossing below zero = bearish momentum shift
        trix_bullish_cross = trix[i] > 0 and trix[i-1] <= 0
        trix_bearish_cross = trix[i] < 0 and trix[i-1] >= 0
        
        # === HTF TREND: 12h EMA(21) direction ===
        htf_trend_up = close[i] > ema_aligned[i]
        htf_trend_down = close[i] < ema_aligned[i]
        
        # === VOLUME CONFIRMATION (1.5x threshold) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX bullish cross + HTF trend up + volume spike ===
            if trix_bullish_cross and htf_trend_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: TRIX bearish cross + HTF trend down + volume spike ===
            elif trix_bearish_cross and htf_trend_down and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Long: exit if price falls 2.5 ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips to down
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short: exit if price rises 2.5 ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips to up
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals