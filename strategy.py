#!/usr/bin/env python3
"""
Experiment #025: 4h TRIX Momentum Breakout + Choppiness Regime + Volume

HYPOTHESIS: TRIX (Triple EMA smoothed ROC) is a leading momentum indicator
that catches reversals BEFORE price breaks. Combined with Choppiness Index
regime filter (avoid ranging markets) and volume confirmation.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: TRIX crosses above 0 + price above 1d SMA50 + volume spike = momentum entry
- Bear: TRIX crosses below 0 + price below 1d SMA50 + volume spike = short entry
- Choppiness Index > 61.8 = ranging = skip (no entry)
- ATR-based stoploss protects against sudden reversals (2022 crash)

KEY INSIGHT FROM DB: TRIX strategies showed ETHUSDT test Sharpe 1.32
CRITICAL: Need 4h HTF trend filter from 1d to reduce whipsaws.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_momentum_chop_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_trix(close, period=15):
    """Triple EMA Smoothed Rate of Change"""
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # ROC of triple-smoothed EMA
    trix = ema3.pct_change(period) * 100  # percentage
    return trix.values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    - CHOP > 61.8 = ranging market (no trend)
    - CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j], abs(low[j] - close[j-1]) if j > 0 else 0)
            atr_sum += tr
        
        if atr_sum <= 0:
            continue
            
        # Highest - Lowest over period
        high_range = np.max(high[i - period + 1:i + 1]) - np.min(low[i - period + 1:i + 1])
        
        if high_range <= 0:
            continue
            
        chop[i] = 100 * np.log10(atr_sum / high_range) / np.log10(period)
    
    return chop

def calculate_sma(values, period):
    """Simple Moving Average"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend (aligned to 4h)
    sma50_1d = calculate_sma(df_1d['close'].values, 50)
    sma50_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # TRIX(15) - momentum
    trix = calculate_trix(close, period=15)
    
    # TRIX signal line (EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Choppiness Index(14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    # TRIX crossover tracking
    prev_trix = 0.0
    prev_trix_signal = 0.0
    
    warmup = 100  # Need enough for TRIX, SMA50_1d alignment, Choppiness
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(sma50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: Choppiness ===
        # Skip if ranging (CHOP > 61.8)
        if chop[i] > 61.8:
            if in_position:
                # Keep existing position but don't add
                signals[i] = position_side * SIZE
            else:
                signals[i] = 0.0
            continue
        
        # === TREND DIRECTION from 1d SMA50 ===
        price_vs_1d = close[i] > sma50_aligned[i]
        bear_trend_1d = close[i] < sma50_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === TRIX MOMENTUM SIGNALS ===
        curr_trix = trix[i]
        curr_signal = trix_signal[i]
        
        # Bullish crossover: TRIX crosses above signal line
        bullish_cross = (prev_trix <= prev_trix_signal and curr_trix > curr_signal)
        
        # Bearish crossover: TRIX crosses below signal line  
        bearish_cross = (prev_trix >= prev_trix_signal and curr_trix < curr_signal)
        
        # Update previous
        prev_trix = curr_trix
        prev_trix_signal = curr_signal
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish TRIX crossover + bull trend (above 1d SMA50) + volume spike
            # TRENDING market (CHOP < 61.8 already checked)
            if bullish_cross and price_vs_1d and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Bearish TRIX crossover + bear trend (below 1d SMA50) + volume spike
            elif bearish_cross and bear_trend_1d and vol_spike:
                desired_signal = -SIZE
        
        # === EXIT / STOP LOGIC ===
        if in_position:
            if position_side > 0:
                # Trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.5 ATR from highest
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                else:
                    desired_signal = SIZE
                
                # Exit if momentum flips (TRIX goes negative)
                if trix[i] < 0 and trix[i-1] >= 0:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.5 ATR from lowest
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                else:
                    desired_signal = -SIZE
                
                # Exit if momentum flips (TRIX goes positive)
                if trix[i] > 0 and trix[i-1] <= 0:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0 and (not in_position or np.sign(desired_signal) != position_side):
            in_position = True
            position_side = int(np.sign(desired_signal))
            entry_price = close[i]
            entry_atr = atr_14[i]
            entry_bar = i
            trailing_high = high[i]
            trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals