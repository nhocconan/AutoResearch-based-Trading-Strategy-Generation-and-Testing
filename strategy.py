#!/usr/bin/env python3
"""
EXPERIMENT #086 - Bollinger Mean Reversion + RSI + 4h Trend Filter (30m primary)
=================================================================================
Hypothesis: 30m Bollinger Band mean reversion works well when filtered by 4h trend.
Unlike pure trend-following (which failed in experiments #074-#085), mean reversion
on 30m captures oscillations within the larger 4h trend. RSI confirms extremes,
volume filter reduces false signals. This differs from previous attempts by:
- Using mean reversion (not trend-following) on 30m
- 4h HMA(21) trend filter (not 1d/1w which are too slow for 30m entries)
- BB touch + RSI extreme combo (not just RSI pullback which failed)
- Volume confirmation filter (avg volume ratio > 1.2)

Key features:
- Primary TF: 30m
- HTF filter: 4h HMA(21) for trend direction
- Entry: BB touch + RSI extreme (35/65) + volume confirmation
- Exit: RSI cross 50 or BB middle or trend reversal
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R, trail stop at 1R

Why this should beat current best (Sharpe=0.490):
- Mean reversion on 30m captures more frequent opportunities than 12h trend
- 4h HMA filter prevents trading against major trend (reduces DD)
- BB + RSI combo more robust than RSI alone (which failed in #074-#085)
- Conservative sizing (0.25-0.30) with proper stoploss controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "bb_rsi_meanreversion_30m_4h_v1"
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


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    middle = sma
    return upper, middle, lower


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period, n):
        if loss_smooth[i] > 0:
            rs[i] = gain_smooth[i] / loss_smooth[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_avg
    ratio[vol_avg == 0] = 1.0
    return ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size
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
        # Check for NaN or zero in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_4h_trend = 1 if price_above_4h_hma else -1
        
        # Bollinger Band touch detection
        bb_lower_touch = low[i] <= bb_lower[i] * 1.001  # Within 0.1% of lower band
        bb_upper_touch = high[i] >= bb_upper[i] * 0.999  # Within 0.1% of upper band
        
        # RSI extremes
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Volume confirmation (above average)
        volume_confirmed = vol_ratio[i] > 1.2
        
        # Determine target signal based on all filters
        target_signal = 0.0
        position_size = BASE_SIZE
        
        # Long entry: 4h bullish + BB lower touch + RSI oversold + volume
        if (hma_4h_trend == 1 and bb_lower_touch and rsi_oversold and volume_confirmed):
            target_signal = position_size
        
        # Short entry: 4h bearish + BB upper touch + RSI overbought + volume
        elif (hma_4h_trend == -1 and bb_upper_touch and rsi_overbought and volume_confirmed):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
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
                # Maintain existing position - check exit conditions
                # Exit if RSI crosses back through 50 (mean reversion complete)
                rsi_exit_long = rsi[i] > 50 and position_side == 1
                rsi_exit_short = rsi[i] < 50 and position_side == -1
                
                # Exit if price reaches BB middle (mean reversion target)
                bb_middle_exit_long = close[i] >= bb_middle[i] and position_side == 1
                bb_middle_exit_short = close[i] <= bb_middle[i] and position_side == -1
                
                # Exit if 4h trend reverses against position
                hma_reversal = (position_side == 1 and hma_4h_trend == -1) or \
                               (position_side == -1 and hma_4h_trend == 1)
                
                if rsi_exit_long or rsi_exit_short or bb_middle_exit_long or \
                   bb_middle_exit_short or hma_reversal:
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