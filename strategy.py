#!/usr/bin/env python3
"""
EXPERIMENT #035 - KAMA Donchian Breakout with Dual HTF Filter (12h primary, 1d/1w HTF)
======================================================================================
Hypothesis: 12h timeframe captures medium-term trends with fewer false signals than 
lower timeframes. KAMA (Kaufman Adaptive MA) adapts to market efficiency - fast in 
trends, slow in chop. Donchian breakout provides clear entry/exit points. Dual HTF 
filter (1d HMA + 1w HMA) ensures we only trade with major trend alignment. Volume 
confirmation on breakouts reduces false signals. This differs from previous KAMA 
attempts by using Donchian entries instead of crossovers, and adding 1w HTF filter.

Key features:
- Primary TF: 12h (MANDATORY for this experiment)
- HTF filters: 1d HMA(50) + 1w HMA(21) for triple trend alignment
- Trend: KAMA(21) on 12h with Efficiency Ratio filter
- Entry: Donchian(20) breakout in trend direction
- Confirmation: Volume > 1.5x 20-period average on breakout bar
- Regime: ADX(14) > 25 for trending market only
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 discrete (25% of capital), reduce to 0.125 at 2R profit
- Take profit: Trail stop at 1R, exit at trend reversal
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_donchian_dualhtf_volume_12h_1d_1w_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise - fast in trends, slow in chop
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    signal = abs(close_s - close_s.shift(period))
    noise = abs(close_s - close_s.shift(1))
    for i in range(1, period):
        noise = noise + abs(close_s.shift(i) - close_s.shift(i + 1))
    
    er = signal / (noise + 1e-10)
    er = er.fillna(0)
    
    # Calculate smoothing constants
    fast_sc = (2 / (fast_period + 1)) ** 2
    slow_sc = (2 / (slow_period + 1)) ** 2
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i - 1] + sc.iloc[i] * (close[i] - kama[i - 1])
    
    return kama, er.values


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    upper[:period - 1] = np.nan
    lower[:period - 1] = np.nan
    
    return upper, lower


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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (tr_smooth + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (tr_smooth + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only - Rule 2)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    kama, er = calculate_kama(close, period=21)
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume moving average for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - discrete level)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(vol_sma[i]) or
            atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HTF trend alignment (12h price vs 1d HMA vs 1w HMA)
        trend_12h = 1 if close[i] > kama[i] else -1
        trend_1d = 1 if close[i] > hma_1d_aligned[i] else -1
        trend_1w = 1 if close[i] > hma_1w_aligned[i] else -1
        
        # ADX regime filter (trending market only)
        regime_valid = adx[i] > 25
        
        # Volume confirmation (breakout volume > 1.5x average)
        volume_confirmed = volume[i] > 1.5 * vol_sma[i]
        
        # Efficiency Ratio filter (KAMA adapts - only trade when ER > 0.3)
        er_valid = er[i] > 0.3
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: All trends bullish + breakout + volume + regime + ER
        if (trend_12h == 1 and trend_1d == 1 and trend_1w == 1 and
            breakout_long and volume_confirmed and regime_valid and er_valid):
            target_signal = SIZE
        
        # Short entry: All trends bearish + breakout + volume + regime + ER
        elif (trend_12h == -1 and trend_1d == -1 and trend_1w == -1 and
              breakout_short and volume_confirmed and regime_valid and er_valid):
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal_exit = False
        
        if position_side != 0:
            r_value = 2.5 * entry_atr  # R = 2.5*ATR at entry
            
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 2 * r_value:
                        take_profit_triggered = True
                
                # Check trend reversal exit (KAMA cross)
                if trend_12h == -1:
                    trend_reversal_exit = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    if close[i] <= entry_price - 2 * r_value:
                        take_profit_triggered = True
                
                # Check trend reversal exit (KAMA cross)
                if trend_12h == 1:
                    trend_reversal_exit = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        elif trend_reversal_exit:
            # Exit on trend reversal
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
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
            elif position_side != 0:
                # Maintain existing position
                if profit_target_hit:
                    signals[i] = HALF_SIZE * position_side
                else:
                    signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals