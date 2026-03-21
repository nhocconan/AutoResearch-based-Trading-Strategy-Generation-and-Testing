#!/usr/bin/env python3
"""
EXPERIMENT #071 - DEMA Crossover + ATR Volatility Regime + Dual HTF Filter (12h primary)
========================================================================================
Hypothesis: DEMA (Double EMA) provides faster response than EMA but less noise than HMA.
On 12h timeframe, DEMA(8)/DEMA(21) crossover captures trend changes early. ATR volatility
regime filter ensures we only trade when volatility is in normal range (not extreme squeeze
or expansion). Dual HTF (1d/1w HMA) confirms major trend direction.

Key features:
- Primary TF: 12h
- Entry: DEMA(8) crosses DEMA(21) with volume confirmation
- Regime: ATR(14) percentile between 30th-70th (avoid extremes)
- HTF filters: 1d HMA(50) + 1w HMA(50) for trend alignment
- Stoploss: 2.5*ATR(14) trailing stop
- Take profit: Reduce to half at 2.5R, trail stop at 1.5R
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30)

Why this should beat current best (Sharpe=0.490):
- DEMA responds faster to trend changes than HMA/EMA
- ATR regime filter avoids trading in extreme volatility (whipsaws)
- 12h timeframe reduces noise vs lower TFs
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dema_atrregime_dualhtf_12h_1d_1w_v1"
timeframe = "12h"
leverage = 1.0


def calculate_dema(close, period):
    """Calculate Double Exponential Moving Average"""
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    dema = 2 * ema1 - ema2
    return dema.values


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
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


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]):
            window_data = series[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= series[i]) / len(window_data)
    
    return pr


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.ewm(span=period, adjust=False, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    dema_fast = calculate_dema(close, 8)
    dema_slow = calculate_dema(close, 21)
    atr = calculate_atr(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Calculate ATR percentile rank (volatility regime filter)
    atr_pr = calculate_percentile_rank(atr, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
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
            np.isnan(dema_fast[i]) or np.isnan(dema_slow[i]) or
            np.isnan(atr[i]) or np.isnan(atr_pr[i]) or np.isnan(vol_ma[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend alignment
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # HTF trend direction
        daily_trend = 1 if price_above_1d_hma else -1
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # ATR volatility regime filter (only trade in normal volatility)
        # Avoid extreme squeeze (<30th) and extreme expansion (>70th)
        atr_normal = 0.30 <= atr_pr[i] <= 0.70
        
        # Volume confirmation (volume > 20-period MA)
        volume_confirmed = volume[i] > vol_ma[i]
        
        # DEMA crossover signals
        dema_bullish = dema_fast[i] > dema_slow[i]
        dema_bearish = dema_fast[i] < dema_slow[i]
        
        # DEMA crossover detection (fast crosses above/below slow)
        dema_cross_long = (dema_fast[i] > dema_slow[i]) and (dema_fast[i-1] <= dema_slow[i-1])
        dema_cross_short = (dema_fast[i] < dema_slow[i]) and (dema_fast[i-1] >= dema_slow[i-1])
        
        # Calculate position size based on ATR regime
        if atr_normal:
            position_size = BASE_SIZE
        else:
            position_size = MIN_SIZE  # Reduce size in extreme volatility
        
        position_size = min(MAX_SIZE, max(MIN_SIZE, position_size))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: DEMA cross long + ATR normal + volume confirmed + HTF bullish
        if (dema_cross_long and atr_normal and volume_confirmed and
            daily_trend == 1 and weekly_trend == 1):
            target_signal = position_size
        
        # Short entry: DEMA cross short + ATR normal + volume confirmed + HTF bearish
        elif (dema_cross_short and atr_normal and volume_confirmed and
              daily_trend == -1 and weekly_trend == -1):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2.5R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
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
                # Maintain existing position (check if trend reversed)
                # Exit if DEMA reverses OR HTF alignment breaks
                dema_reversal_long = dema_bearish
                dema_reversal_short = dema_bullish
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                
                if dema_reversal_long or dema_reversal_short or hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals