#!/usr/bin/env python3
"""
EXPERIMENT #010 - Donchian Breakout + KAMA Trend + 1d HMA Filter (4h primary)
=====================================================================================
Hypothesis: 4h Donchian breakouts capture major trend moves in crypto, but generate false
signals in ranging markets. Adding 1d HMA(21) trend filter ensures we only trade breakouts
in the direction of the daily trend. KAMA(14) adapts to volatility - flat KAMA = avoid trades.
Volume confirmation (1.5x average) ensures breakout has conviction.

Key features:
- Primary TF: 4h (required for Experiment #010)
- HTF filter: 1d HMA(21) for major trend direction
- Entry: Donchian(20) breakout with volume confirmation
- Trend filter: KAMA(14) slope + 1d HMA alignment
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, max 0.35 (conservative to control DD)
- Take profit: Reduce to half at 2.5R, trail stop at 1.5R

Why this should work:
- 4h captures major moves without 15m noise (failed in Exp #001, #007)
- Donchian breakout is simpler than Supertrend+RSI+ADX (Exp #004 had 0 trades)
- Volume filter reduces false breakouts
- KAMA adapts to volatility better than static EMA/HMA
- Conservative sizing (0.25-0.35) prevents -80% DD seen in Exp #008, #009
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_kama_1dhma_4h_v1"
timeframe = "4h"
leverage = 1.0


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


def calculate_kama(close, period=14, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)"""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period - 1, n):
        signal = abs(close[i] - close[i - period + 1])
        noise = 0.0
        for j in range(i - period + 2, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    for i in range(period - 1, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    # Calculate KAMA
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = high[i - period + 1:i + 1].max()
        lower[i] = low[i - period + 1:i + 1].min()
    
    return upper, lower


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, period=14)
    kama = calculate_kama(close, period=14, fast=2, slow=30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.35   # Max position size with strong volume
    MIN_SIZE = 0.15   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    trailing_active = False
    
    min_period = 50  # Wait for indicators to stabilize (less strict than 100)
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(atr[i]) or np.isnan(donchian_upper[i]) or
            np.isnan(vol_ma[i]) or atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1d HMA trend filter
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        daily_trend = 1 if price_above_1d_hma else -1
        
        # KAMA slope (trend direction on 4h)
        kama_slope = 0
        if i >= 3 and not np.isnan(kama[i - 3]):
            kama_slope = 1 if kama[i] > kama[i - 3] else -1
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # Volume confirmation (1.5x average volume)
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Calculate position size based on volume strength (dynamic sizing)
        vol_multiplier = min(1.0 + (volume[i] / vol_ma[i] - 1.5) / 2, 1.4)  # Max 1.4x
        vol_multiplier = max(0.8, vol_multiplier)  # Min 0.8x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Donchian breakout + 1d HMA bullish + KAMA slope up + Volume confirmed
        if breakout_long and daily_trend == 1 and kama_slope == 1 and volume_confirmed:
            target_signal = position_size
        
        # Short entry: Donchian breakout + 1d HMA bearish + KAMA slope down + Volume confirmed
        elif breakout_short and daily_trend == -1 and kama_slope == -1 and volume_confirmed:
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                
                # Trailing stop logic
                if not trailing_active:
                    # Activate trailing stop at 1.5R profit
                    if close[i] >= entry_price + 3.0 * entry_atr:
                        trailing_active = True
                
                if trailing_active:
                    # Trail at highest - 2.5*ATR
                    trailing_stop = highest_since_entry - 2.5 * atr[i]
                else:
                    # Initial stop at entry - 2.5*ATR
                    trailing_stop = entry_price - 2.5 * entry_atr
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2.5R from entry)
                if not profit_target_hit and not trailing_active:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2.5R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                
                # Trailing stop logic
                if not trailing_active:
                    # Activate trailing stop at 1.5R profit
                    if close[i] <= entry_price - 3.0 * entry_atr:
                        trailing_active = True
                
                if trailing_active:
                    # Trail at lowest + 2.5*ATR
                    trailing_stop = lowest_since_entry + 2.5 * atr[i]
                else:
                    # Initial stop at entry + 2.5*ATR
                    trailing_stop = entry_price + 2.5 * entry_atr
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit and not trailing_active:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2.5R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            trailing_active = False
        elif take_profit_triggered:
            # Reduce position to half at 2.5R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            trailing_active = True  # Activate trailing after TP
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
                trailing_active = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if daily trend reverses OR KAMA slope reverses
                daily_reversal = (position_side == 1 and daily_trend == -1) or \
                                 (position_side == -1 and daily_trend == 1)
                kama_reversal = (position_side == 1 and kama_slope == -1) or \
                                (position_side == -1 and kama_slope == 1)
                
                if daily_reversal or kama_reversal:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                    trailing_active = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals