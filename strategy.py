#!/usr/bin/env python3
"""
Experiment #023: Simple Donchian Breakout + Volume + ATR Stop (4h)

HYPOTHESIS: Simple price channel breakout with volume confirmation and 
fixed ATR stop generates consistent entries with manageable drawdown.

WHY IT SHOULD WORK:
- Donchian(20) breakout captures momentum after consolidation
- Volume spike filters false breakouts
- ATR-based stoploss manages risk symmetrically
- Simple = more trades = better statistical validity
- This pattern achieved test Sharpe 1.38-1.49 on SOLUSDT (DB verified)

EXPECTED TRADE COUNT: 75-200 total over 4 years
- Donchian breaks ~1-3x per week on 4h → 52-156/year potential
- Volume filter (1.5x) → 50% reduction → 26-78/year
- Some additional filtering → ~25-75 trades/year
- Final: ~100-300 total = statistical validity
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_atr_simple_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF EMA for trend (1d) ===
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) - highest high and lowest low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 periods)
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
    
    warmup = 250  # Donchian(20) + vol_ma(20) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i-1]) or np.isnan(donchian_lower[i-1]):
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # HTF trend (from 1d)
        htf_bull = close[i] > ema50_1d_aligned[i] if not np.isnan(ema50_1d_aligned[i]) else False
        htf_bear = close[i] < ema50_1d_aligned[i] if not np.isnan(ema50_1d_aligned[i]) else False
        
        if not in_position:
            # === LONG ENTRY: Price breaks above Donchian high + volume spike ===
            bullish_breakout = high[i] > donchian_upper[i-1]
            
            if bullish_breakout and vol_spike and htf_bull:
                desired_signal = SIZE
                
            # === SHORT ENTRY: Price breaks below Donchian low + volume spike ===
            bearish_breakout = low[i] < donchian_lower[i-1]
            
            if bearish_breakout and vol_spike and htf_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Stop: price falls 2 ATR below entry
                stop_price = entry_price - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Exit if trend reverses (close below HTF EMA)
                elif close[i] < ema50_1d_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Stop: price rises 2 ATR above entry
                stop_price = entry_price + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                # Exit if trend reverses (close above HTF EMA)
                elif close[i] > ema50_1d_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        
        signals[i] = desired_signal
    
    return signals