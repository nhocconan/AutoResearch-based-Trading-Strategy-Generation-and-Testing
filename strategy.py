#!/usr/bin/env python3
"""
Experiment #024: 12h Donchian + 1d EMA Trend + Volume Confirmation

HYPOTHESIS: 12h timeframe with 1d HTF trend filter captures medium-term swings
while filtering noise. 1d EMA provides cleaner trend direction than 12h EMA alone.

WHY IT SHOULD WORK IN BOTH MARKETS:
- 12h captures swing trades (2-5 day holds) vs 4h choppiness
- 1d EMA smooths noise better than lower timeframes
- Donchian(8) on 12h = same as Donchian(16) on 6h - proven structural element
- Volume spike confirms institutional interest at breakouts
- 2.5 ATR stop handles 2022 volatility

TRADE COUNT ESTIMATE:
- 12h bars over 4 years: ~14,600 bars
- 1d EMA filter: ~50% of bars
- Donchian(8) breakout: ~10% pass rate
- Volume 1.3x confirmation: ~60% pass
- Final: ~0.5-0.8% of bars = 70-115 trades/4yr
- Target: 75-100 trades per symbol (within 50-150 target range)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for trend direction (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(8) for breakout structure
    donchian_upper = pd.Series(high).rolling(window=8, min_periods=8).max().values
    donchian_lower = pd.Series(low).rolling(window=8, min_periods=8).min().values
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # Need for EMA(21) on 1d HTF + local indicators
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === HTF TREND DIRECTION (1d EMA aligned) ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] >= 1.3
        
        # === DONCHIAN BREAKOUT (prior bar's range for no look-ahead) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior bar's upper channel
        bullish_breakout = not np.isnan(prev_upper) and close[i] > prev_upper
        
        # Bearish breakout: close below prior bar's lower channel
        bearish_breakout = not np.isnan(prev_lower) and close[i] < prev_lower
        
        # === MINIMUM HOLD: 2 bars ===
        min_hold = (i - entry_bar) >= 2
        
        # === POSITION MANAGEMENT ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_hit = low[i] < (entry_price - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (entry_price + 2.5 * entry_atr)
            
            # Exit on opposite breakout (trend reversal)
            reversal_exit = (position_side > 0 and bearish_breakout) or \
                           (position_side < 0 and bullish_breakout)
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold and reversal_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Need HTF alignment + breakout + volume
            # LONG: HTF bullish + bullish breakout + volume spike
            if htf_bullish and bullish_breakout and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: HTF bearish + bearish breakout + volume spike
            elif htf_bearish and bearish_breakout and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals