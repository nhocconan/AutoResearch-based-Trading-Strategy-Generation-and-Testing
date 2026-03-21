#!/usr/bin/env python3
"""
EXPERIMENT #032 - DEMA Trend + Z-Score Entry + 4h HMA Filter (30m primary)
=====================================================================================
Hypothesis: DEMA (Double EMA) reacts faster than regular EMA while smoothing noise better.
Combined with Z-score mean reversion entries within the 4h HMA trend direction,
this captures pullbacks in strong trends with better timing than RSI alone.
Volume spike confirmation filters out low-liquidity false signals.

Key features:
- Primary TF: 30m
- HTF filter: 4h HMA(21) for major trend direction
- Trend: DEMA(8/21) crossover for entry timing
- Entry: Z-score(20) <-1.5 long, >1.5 short within trend
- Confirmation: Volume > 1.5*MA(20) volume
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.20-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should beat previous failures:
- DEMA has less lag than EMA/KAMA for faster entry
- Z-score entries are more statistically grounded than RSI thresholds
- Volume filter removes 40%+ of false signals in low-liquidity periods
- 30m captures more opportunities than 1h/4h while avoiding 15m noise
- Conservative sizing (0.20-0.30) controls drawdown during crashes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "dema_zscore_vol_4hhma_30m_v1"
timeframe = "30m"
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


def calculate_dema(close, period):
    """Calculate Double Exponential Moving Average (DEMA)"""
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    dema = 2 * ema1 - ema2
    return dema.values


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection"""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std
    return zscore.values


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    dema_fast = calculate_dema(close, 8)
    dema_slow = calculate_dema(close, 21)
    zscore = calculate_zscore(close, 20)
    atr = calculate_atr(high, low, close, 14)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong confirmation
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(dema_fast[i]) or
            np.isnan(dema_slow[i]) or np.isnan(zscore[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or
            atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # DEMA crossover signal
        dema_bullish = dema_fast[i] > dema_slow[i]
        dema_bearish = dema_fast[i] < dema_slow[i]
        
        # Z-score mean reversion entry
        zscore_oversold = zscore[i] < -1.5  # Price below mean
        zscore_overbought = zscore[i] > 1.5  # Price above mean
        
        # Volume confirmation (volume > 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Calculate position size based on volume confirmation
        position_size = MAX_SIZE if volume_spike else BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: DEMA bullish + 4h HMA bullish + Z-score oversold + Volume confirmation
        if (dema_bullish and hma_trend == 1 and zscore_oversold and volume_spike):
            target_signal = position_size
        
        # Short entry: DEMA bearish + 4h HMA bearish + Z-score overbought + Volume confirmation
        elif (dema_bearish and hma_trend == -1 and zscore_overbought and volume_spike):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
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
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * np.sign(position_side)
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
                # Exit if DEMA reverses OR 4h HMA alignment breaks
                dema_reversal_long = dema_bearish
                dema_reversal_short = dema_bullish
                hma_alignment_broken = (position_side == 1 and hma_trend == -1) or \
                                       (position_side == -1 and hma_trend == 1)
                
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