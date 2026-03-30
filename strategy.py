#!/usr/bin/env python3
"""
Experiment #025: 4h TRIX Momentum + Volume Spike + Weekly ATR Regime (4h)

HYPOTHESIS: TRIX(28) for trend momentum, volume spike confirmation,
weekly ATR regime filter to avoid low-vol whipsaws.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: TRIX crosses above 0 + volume spike + rising ATR = momentum entry
- Bear: TRIX crosses below 0 + volume spike + rising ATR = bearish entry
- Weekly ATR filter: only enter when volatility is expanding (not during chop)

EXPECTED TRADES: 100-180 total over 4 years (25-45/year per symbol)
- TRIX crossover on 4h = signal every ~30-50 bars
- Volume spike 1.8x → reduces by ~40%
- Weekly ATR rising filter → reduces by ~30%
- Final: ~100-180 trades = statistical validity without overtrading
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_vol_atr_regime_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=28):
    """Triple EMA - TRIX momentum indicator"""
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    trix = ema3.pct_change() * 100
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

def calculate_weekly_atr(df_1w):
    """Weekly ATR for regime filter"""
    high = df_1w['high'].values
    low = df_1w['low'].values
    close = df_1w['close'].values
    n = len(close)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Weekly ATR(14) = 14-week EMA of TR
    atr_weekly = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    return atr_weekly

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    weekly_atr = calculate_weekly_atr(df_1w)
    weekly_atr_aligned = align_htf_to_ltf(prices, df_1w, weekly_atr)
    
    # Weekly EMA for trend direction
    weekly_ema = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix = calculate_trix(close, period=28)
    
    # Local ATR for comparison
    local_atr = calculate_atr(high, low, close, period=14)
    
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
    
    warmup = 80  # Enough for TRIX(28), ATR(14), volume(20), weekly alignment
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(trix[i]) or np.isnan(weekly_atr_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND FILTER ===
        weekly_trend_up = close[i] > weekly_ema_aligned[i]
        weekly_trend_down = close[i] < weekly_ema_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === WEEKLY ATR REGIME: only trade when volatility is rising ===
        # Compare current 4h ATR to aligned weekly ATR (scaled down)
        atr_ratio = local_atr[i] / (weekly_atr_aligned[i] / 24)  # Scale weekly to 4h equivalent
        vol_expanding = atr_ratio > 0.7  # Vol is expanding or at least not compressed
        
        # === TRIX CROSSOVER SIGNALS ===
        trix_curr = trix[i]
        trix_prev = trix[i-1] if i > 0 and not np.isnan(trix[i-1]) else 0.0
        
        # Bullish crossover: TRIX crosses above 0
        bullish_cross = (trix_prev <= 0) and (trix_curr > 0)
        # Bearish crossover: TRIX crosses below 0
        bearish_cross = (trix_prev >= 0) and (trix_curr < 0)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: TRIX bullish cross + volume spike + vol expanding + weekly uptrend
            if bullish_cross and vol_spike and vol_expanding and weekly_trend_up:
                desired_signal = SIZE
            
            # SHORT: TRIX bearish cross + volume spike + vol expanding + weekly downtrend
            elif bearish_cross and vol_spike and vol_expanding and weekly_trend_down:
                desired_signal = -SIZE
        
        # === EXIT LOGICS ===
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
                
                # Exit on TRIX reversal or trend flip
                elif trix_curr < 0 or close[i] < weekly_ema_aligned[i]:
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
                
                # Exit on TRIX reversal or trend flip
                elif trix_curr > 0 or close[i] > weekly_ema_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 6 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals