#!/usr/bin/env python3
"""
EXPERIMENT #019 - HMA Cross + RSI + BB Regime + 4h Trend Filter (15m primary)
=====================================================================================
Hypothesis: Previous strategies failed due to too many conflicting filters (ADX+DI+RSI+Supertrend)
resulting in either 0 trades or massive drawdowns from position sizing issues.

This strategy simplifies entry logic while maintaining strict risk control:
- Primary TF: 15m HMA(9)/HMA(21) crossover for fast trend detection
- HTF Filter: 4h HMA(21) slope (only trade with 4h trend direction)
- Regime Filter: Bollinger Band Width percentile (avoid low volatility chop)
- Entry Confirmation: RSI(14) not extreme (>30 for long, <70 for short)
- Volume: 20-bar volume MA confirmation (>1.0x average)
- Stoploss: 2.0*ATR(14) trailing stop
- Position sizing: 0.20-0.25 discrete (CONSERVATIVE to avoid -85% DD)

Why this should work:
- HMA is faster than EMA, catches moves earlier on 15m
- 4h HMA slope filter removes 50% of false signals
- BB Width regime filter avoids choppy periods (major DD source)
- Conservative sizing (0.20-0.25) prevents blowup during 2022-style crashes
- Simpler logic = more trades (avoid 0-trade failure mode)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_cross_rsi_bbregime_4h_15m_v1"
timeframe = "15m"
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


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma * 100  # Bandwidth as percentage
    return upper, lower, bb_width


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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    hma_fast = calculate_hma(close, 9)
    hma_slow = calculate_hma(close, 21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    # Calculate BB Width percentile for regime filter (rolling 100 bars)
    bb_width_percentile = np.zeros(n)
    bb_width_percentile[:] = np.nan
    lookback = 100
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            valid_widths = bb_width[i-lookback:i+1]
            valid_widths = valid_widths[~np.isnan(valid_widths)]
            if len(valid_widths) > 0:
                bb_width_percentile[i] = np.sum(bb_width[i] >= valid_widths) / len(valid_widths) * 100
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.22  # Conservative base position size (22% of capital)
    MAX_SIZE = 0.28   # Max position size
    MIN_SIZE = 0.18   # Min position size
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
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(bb_width[i]) or
            np.isnan(vol_ma[i]) or np.isnan(bb_width_percentile[i]) or
            atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA slope for trend direction (compare current vs 5 bars ago)
        hma_4h_slope = 0
        if i >= 5:
            if hma_4h_aligned[i] > hma_4h_aligned[i-5]:
                hma_4h_slope = 1  # Bullish
            elif hma_4h_aligned[i] < hma_4h_aligned[i-5]:
                hma_4h_slope = -1  # Bearish
        
        # 15m HMA crossover signal
        hma_cross = 0
        if hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]:
            hma_cross = 1  # Bullish cross
        elif hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]:
            hma_cross = -1  # Bearish cross
        
        # Current HMA alignment
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # RSI filter (avoid extremes)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 70  # Not oversold for long entry
        rsi_ok_short = rsi[i] > 30 and rsi[i] < 70  # Not overbought for short entry
        
        # Volume confirmation (>1.0x 20-bar average)
        volume_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirmed = volume_ratio > 1.0
        
        # BB Width regime filter (avoid low volatility chop)
        # Only trade when BB Width is in top 40% of recent range (volatility expanding)
        regime_ok = bb_width_percentile[i] > 40
        
        # Calculate position size (conservative)
        position_size = BASE_SIZE
        
        # Determine target signal based on filters
        target_signal = 0.0
        
        # Long entry: HMA bullish + 4h slope bullish + RSI ok + volume confirmed + regime ok
        if (hma_bullish and hma_4h_slope == 1 and rsi_ok_long and 
            volume_confirmed and regime_ok):
            target_signal = position_size
        
        # Short entry: HMA bearish + 4h slope bearish + RSI ok + volume confirmed + regime ok
        elif (hma_bearish and hma_4h_slope == -1 and rsi_ok_short and 
              volume_confirmed and regime_ok):
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
                # Exit if HMA crosses against position OR 4h slope reverses
                hma_reversal_long = hma_bearish  # Fast crossed below slow
                hma_reversal_short = hma_bullish  # Fast crossed above slow
                hma_slope_reversal = (position_side == 1 and hma_4h_slope == -1) or \
                                     (position_side == -1 and hma_4h_slope == 1)
                
                if hma_reversal_long or hma_reversal_short or hma_slope_reversal:
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