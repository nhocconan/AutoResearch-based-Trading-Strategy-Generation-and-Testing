#!/usr/bin/env python3
"""
EXPERIMENT #020 - Donchian Breakout + Volume + Volatility Regime (30m primary, 4h HTF)
======================================================================================
Hypothesis: Donchian channel breakouts capture trend momentum, but most breakouts fail
in choppy markets. Adding volume confirmation (breakout volume > 1.5x avg) filters
false breakouts. 4h HMA(50) provides major trend alignment. ATR percentile regime
filter ensures we only trade when volatility is in normal range (20-80th percentile),
avoiding extreme chop or panic conditions. This differs from previous attempts by
focusing on breakout confirmation rather than pullback entries.

Key features:
- Primary TF: 30m
- HTF filter: 4h HMA(50) for major trend direction
- Entry: Donchian(20) breakout with volume confirmation (>1.5x 20-bar avg)
- Regime: ATR(14) percentile 20-80% (normal volatility only)
- Stoploss: 2.5*ATR(14) trailing from entry
- Take profit: Reduce to half at 2R profit
- Position sizing: 0.25-0.30 discrete levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_volume_regime_30m_4h_v1"
timeframe = "30m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_volume_spike(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values


def calculate_atr_percentile(atr, window=100):
    """Calculate rolling percentile rank of ATR"""
    n = len(atr)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(atr[i]):
            window_data = atr[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= atr[i]) / len(window_data)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    vol_ratio = calculate_volume_spike(volume, 20)
    atr_pr = calculate_atr_percentile(atr, 100)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 120  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr_pr[i]) or 
            atr[i] == 0 or donchian_upper[i] == donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (HTF)
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Volatility regime filter: only trade when ATR is in 20-80th percentile
        # (avoid extreme chop <20% or panic >80%)
        regime_valid = 0.20 <= atr_pr[i] <= 0.80
        
        # Volume confirmation: breakout volume > 1.5x average
        volume_confirmed = vol_ratio[i] > 1.5
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h trend bullish + Donchian breakout + Volume confirmed + Regime valid
        if hma_trend == 1 and breakout_long and volume_confirmed and regime_valid:
            target_signal = SIZE
        
        # Short entry: 4h trend bearish + Donchian breakout + Volume confirmed + Regime valid
        elif hma_trend == -1 and breakout_short and volume_confirmed and regime_valid:
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if low[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if high[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if high[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if low[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed on HTF)
                if position_side == 1 and hma_trend == -1:
                    # HTF trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and hma_trend == 1:
                    # HTF trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals