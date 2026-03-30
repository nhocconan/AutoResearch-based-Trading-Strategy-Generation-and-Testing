#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian(15) Breakout + Weekly EMA(21) + Volume (1d)

HYPOTHESIS: 1d timeframe with weekly trend confirmation should work across both
bull and bear markets:
- Bull: Breakout above 15-bar high + volume spike + above weekly EMA21 = momentum
- Bear: Breakdown below 15-bar low + volume spike + below weekly EMA21 = momentum
- Weekly EMA21 provides structural trend without being too lagging

EXPECTED TRADES: 35-60 total over 4 years (9-15/year per symbol)
- Donchian(15) on 1d = break every ~15-30 bars = 12-24 potential/year
- Volume spike (1.5x) → reduces by ~40% = 7-14 signals/year
- Weekly EMA21 filter → reduces by ~25% = 5-11 signals/year
- Final: ~35-55 trades = within target (30-100)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA21 for trend direction
    weekly_ema21 = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(15) - SHIFT by 1 to avoid look-ahead
    donchian_upper = pd.Series(high).rolling(window=15, min_periods=15).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=15, min_periods=15).min().shift(1).values
    
    # Volume average (20 bars for smoothing)
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
    
    warmup = 50  # Enough for Donchian15, ATR14, vol_ma20
    
    for i in range(warmup, n):
        # Sanity checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_ema21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: Weekly EMA21 ===
        bull_trend = close[i] > weekly_ema21_aligned[i]
        bear_trend = close[i] < weekly_ema21_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (using PREVIOUS bar's channel, no look-ahead) ===
        bullish_breakout = high[i] > donchian_upper[i]
        bearish_breakout = low[i] < donchian_lower[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend
            if bullish_breakout and vol_spike and bull_trend:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend
            elif bearish_breakout and vol_spike and bear_trend:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
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
                
                # Trend flip exit
                elif close[i] < weekly_ema21_aligned[i]:
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
                
                # Trend flip exit
                elif close[i] > weekly_ema21_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 5 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 5:
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