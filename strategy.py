#!/usr/bin/env python3
"""
EXPERIMENT #041 - Dual HMA Crossover + Multi-TF Trend + Volume Filter (12h primary)
====================================================================================
Hypothesis: 12h timeframe captures multi-day trends better than 4h (less noise) while 
being more responsive than 1d. Dual HMA crossover (21/50) on 12h provides trend signal, 
while 1d HMA(50) and 1w HMA(21) provide multi-timeframe trend alignment. Volume ratio 
filter ensures we only trade on confirmed moves (volume > 1.5x average). This differs 
from previous 12h attempts by adding dual HTF filters and volume confirmation.

Key features:
- Primary TF: 12h
- HTF filters: 1d HMA(50) + 1w HMA(21) for trend alignment
- Trend signal: HMA(21) vs HMA(50) crossover on 12h
- Volume filter: Current volume > 1.5x 20-period average
- Entry timing: RSI(14) not in extremes (<30 or >70)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dual_hma_mtf_volume_12h_1d_1w_v1"
timeframe = "12h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average using WMA"""
    close_s = pd.Series(close)
    
    def wma(series, span):
        """Weighted Moving Average"""
        return series.ewm(span=span, adjust=False, min_periods=span).mean()
    
    wma1 = wma(close_s, period // 2)
    wma2 = wma(close_s, period)
    raw_hma = 2 * wma1 - wma2
    hma = wma(raw_hma, int(np.sqrt(period)))
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio (current / rolling average)"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMAs
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF data to LTF (auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    hma_fast = calculate_hma(close, 21)
    hma_slow = calculate_hma(close, 50)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2
    
    # Track position state
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Multi-timeframe trend alignment
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        weekly_trend = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # 12h HMA crossover signal
        hma_cross = 1 if hma_fast[i] > hma_slow[i] else -1
        
        # Volume filter (only trade on above-average volume)
        volume_confirmed = vol_ratio[i] > 1.5
        
        # RSI filter (avoid extremes)
        rsi_valid = 30 < rsi[i] < 70
        
        # Determine target signal
        target_signal = 0.0
        
        # Long: All trends aligned + volume confirmed + RSI valid
        if hma_cross == 1 and daily_trend == 1 and weekly_trend == 1 and volume_confirmed and rsi_valid:
            target_signal = SIZE
        
        # Short: All trends aligned + volume confirmed + RSI valid
        elif hma_cross == -1 and daily_trend == -1 and weekly_trend == -1 and volume_confirmed and rsi_valid:
            target_signal = -SIZE
        
        # Stoploss and take profit logic
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:
                        take_profit_triggered = True
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * atr[i]:
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            if target_signal != 0.0 and position_side == 0:
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                if position_side == 1 and hma_cross == -1:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and hma_cross == 1:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals