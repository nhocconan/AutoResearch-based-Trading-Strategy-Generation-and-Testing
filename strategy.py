#!/usr/bin/env python3
"""
Experiment #023: 6h Donchian Breakout + 1d EMA Trend Filter

HYPOTHESIS: Multi-timeframe trend alignment between 1d macro and 6h structure
should capture higher-probability moves while avoiding chop.

WHY IT SHOULD WORK IN BOTH MARKETS:
1. 1d EMA(8) captures the macro trend direction across all conditions
2. 6h Donchian(24) identifies structural breaks (~6 days = good balance for 6h)
3. Only entering in direction of daily trend avoids "trapping" in counter-trend
4. 6h timeframe allows ~4 bars/day → manageable trade frequency

ENTRY CONDITIONS:
- 6h close breaks outside Donchian(24) channel
- Volume > 1.5x 20-bar MA (confirms institutional move)
- 1d close > 1d EMA(8) for longs, < for shorts
- Choppiness < 61.8 (not in extreme chop)

EXIT CONDITIONS:
- Stop: 2.5 ATR from entry
- Reversal: opposite Donchian breakout
- Min hold: 2 bars

TRADE COUNT ESTIMATE:
- 6h bars/4yr ≈ 5840
- Donchian(24) breakout: ~1 per 50-70 bars = ~83-117 raw signals
- 1d EMA alignment filter: ~50% pass = ~42-58
- Volume + CHOP filter: ~40% pass = ~17-23 trades/symbol
- NEED TO LOOSEN: use Donchian(20) instead = ~25-40 trades
- SAFE RANGE: 30-50 trades over 4 years per symbol (on lower end)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range - vectorized"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_chop(high, low, close, period=14):
    """Choppiness Index - regime filter"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        period_high = high[i-period+1:i+1].max()
        period_low = low[i-period+1:i+1].min()
        
        if period_high > period_low:
            sum_tr = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                sum_tr += tr
            
            if period_high != period_low:
                chop[i] = 100 * np.log10(sum_tr / (period_high - period_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Load 1d data for macro trend (ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values.astype(np.float64)
    
    # 1d EMA(8) for trend direction
    ema_1d = pd.Series(close_1d).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    # Align 1d EMA to 6h bars (shift by 1 to avoid look-ahead)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_chop(high, low, close, period=14)
    
    # Donchian Channel(20) - price structure
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 50  # Enough for all indicators
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 1d TREND FILTER ===
        # Price above 1d EMA = bullish macro trend
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # === CHOP FILTER ===
        choppy_market = chop[i] > 61.8
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (prior bar's range) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior bar's upper channel
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        
        # Bearish breakout: close below prior bar's lower channel
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === MINIMUM HOLD: 2 bars ===
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
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
            # Skip if choppy market
            if choppy_market:
                signals[i] = 0.0
                continue
            
            # LONG: Bullish breakout + volume spike + 1d bullish trend
            if bullish_breakout and vol_spike and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume spike + 1d bearish trend
            elif bearish_breakout and vol_spike and bearish_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals