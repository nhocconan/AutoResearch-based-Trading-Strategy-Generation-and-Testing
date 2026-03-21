#!/usr/bin/env python3
"""
EXPERIMENT #013 - Multi-TF Pullback Strategy (15m Primary)
==========================================================
Hypothesis: 15m RSI pullback entries aligned with 4h HMA trend + 1h Supertrend
confirmation will capture trend continuations with better timing than daily-based
strategies. The 15m timeframe provides more entry opportunities while HTF filters
prevent counter-trend trades. Volume confirmation reduces false breakouts.

Key features:
- Primary TF: 15m (more trade opportunities than 1h/4h/1d)
- HTF filter 1: 4h HMA(21) for major trend direction
- HTF filter 2: 1h Supertrend(10,3) for intermediate trend confirmation
- Entry: 15m RSI(14) pullback to 40-60 zone + volume > 20-period average
- Stoploss: 2.5*ATR(14) trailing stop
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this differs from failed strategies:
- Uses 15m primary (not tried successfully in experiment history)
- Triple timeframes: 4h trend + 1h confirmation + 15m entry
- RSI pullback (not breakout) = better risk/reward in trending markets
- Volume filter on 15m (not daily) = more responsive
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_rsi_pullback_4h_1h_15m_v1"
timeframe = "15m"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = uptrend, -1 = downtrend
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = np.nan
            continue
        
        upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
        lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1 if close[i] < supertrend[i] else 1
        else:
            # Update bands based on previous trend
            if trend[i - 1] == 1:
                lower_band[i] = max(lower_band[i], lower_band[i - 1])
                if close[i] < lower_band[i]:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
            else:
                upper_band[i] = min(upper_band[i], upper_band[i - 1])
                if close[i] > upper_band[i]:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
    
    return supertrend, trend


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA(21) for major trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h Supertrend(10, 3) for intermediate trend
    supertrend_1h, trend_1h = calculate_supertrend(
        df_1h['high'].values,
        df_1h['low'].values,
        df_1h['close'].values,
        period=10,
        multiplier=3.0
    )
    trend_1h_aligned = align_htf_to_ltf(prices, df_1h, trend_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Volume moving average (20-period)
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    entry_atr = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(trend_1h_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma[i]) or np.isnan(rsi[i]) or 
            atr[i] == 0 or volume_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (price above HMA = bullish)
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 1h Supertrend confirmation
        st_trend = int(trend_1h_aligned[i]) if not np.isnan(trend_1h_aligned[i]) else 0
        
        # Volume confirmation (must be above 20-period average)
        volume_confirmed = volume[i] > volume_sma[i] * 1.0  # At least average volume
        
        # RSI pullback zone (40-60 for continuation entries)
        rsi_pullback_long = 40 <= rsi[i] <= 60
        rsi_pullback_short = 40 <= rsi[i] <= 60
        
        # RSI momentum confirmation (rising for long, falling for short)
        rsi_momentum_long = rsi[i] > rsi[i - 1] if i > 0 else False
        rsi_momentum_short = rsi[i] < rsi[i - 1] if i > 0 else False
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h uptrend + 1h supertrend up + RSI pullback + volume
        if trend_4h == 1 and st_trend == 1 and rsi_pullback_long and rsi_momentum_long and volume_confirmed:
            target_signal = SIZE
        
        # Short entry: 4h downtrend + 1h supertrend down + RSI pullback + volume
        elif trend_4h == -1 and st_trend == -1 and rsi_pullback_short and rsi_momentum_short and volume_confirmed:
            target_signal = -SIZE
        
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
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit and entry_atr > 0:
                    profit_target = entry_price + 5.0 * entry_atr  # 2R = 5*ATR
                    if close[i] >= profit_target:
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit and entry_atr > 0:
                    profit_target = entry_price - 5.0 * entry_atr  # 2R profit
                    if close[i] <= profit_target:
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
            entry_atr = 0.0
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
                profit_target_hit = False
                entry_atr = atr[i]
            elif position_side != 0:
                # Maintain existing position (check if we should exit on trend reversal)
                if position_side == 1 and (trend_4h == -1 or st_trend == -1):
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                    entry_atr = 0.0
                elif position_side == -1 and (trend_4h == 1 or st_trend == 1):
                    # Trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                    entry_atr = 0.0
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals