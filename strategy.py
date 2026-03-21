#!/usr/bin/env python3
"""
EXPERIMENT #001 - Supertrend + RSI Pullback with 4h Trend Filter (15m)
=======================================================================
Hypothesis: 15m Supertrend entries aligned with 4h HMA(21) trend direction
capture intraday momentum while avoiding counter-trend trades. RSI(14) pullback
filter ensures we enter on dips in uptrends (RSI 40-55) and rallies in downtrends
(RSI 45-60), not at extremes. Volume confirmation filters false breakouts.
ATR(14) trailing stop at 2.5x protects against reversals.

Key features:
- Primary TF: 15m (intraday momentum)
- HTF filter: 4h HMA(21) for major trend direction
- Entry: Supertrend(10,3) flip + RSI pullback zone + volume > SMA(20)
- Filter: 4h trend must align with entry direction
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_4h_filter_15m_v1"
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
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            upper_band[i] = np.nan
            lower_band[i] = np.nan
            supertrend[i] = np.nan
            direction[i] = 0
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            # Upper band logic
            if close[i - 1] <= supertrend[i - 1]:
                upper_band[i] = min(upper_band[i], upper_band[i - 1])
            else:
                upper_band[i] = hl2[i] + multiplier * atr[i]
            
            # Lower band logic
            if close[i - 1] >= supertrend[i - 1]:
                lower_band[i] = max(lower_band[i], lower_band[i - 1])
            else:
                lower_band[i] = hl2[i] - multiplier * atr[i]
            
            # Determine supertrend value and direction
            if close[i] <= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1  # Long signal
            elif close[i] >= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1  # Short signal
            else:
                supertrend[i] = supertrend[i - 1]
                direction[i] = direction[i - 1]
    
    return supertrend, direction


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
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    supertrend, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Volume moving average
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
    
    min_period = 100  # Wait for 4h HMA and indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma[i]) or np.isnan(rsi[i]) or 
            atr[i] == 0 or supertrend_dir[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # Volume confirmation (must be above 20-period average)
        volume_confirmed = volume[i] > volume_sma[i]
        
        # Supertrend direction signal
        st_signal = int(supertrend_dir[i])
        st_prev = int(supertrend_dir[i - 1]) if i > 0 else 0
        
        # Detect Supertrend flip (entry signal)
        st_flip = 0
        if st_signal == 1 and st_prev == -1:
            st_flip = 1  # Long flip
        elif st_signal == -1 and st_prev == 1:
            st_flip = -1  # Short flip
        
        # RSI pullback filter
        # For longs: RSI should be in pullback zone (40-55), not overbought
        # For shorts: RSI should be in rally zone (45-60), not oversold
        rsi_valid_long = 40 < rsi[i] < 55
        rsi_valid_short = 45 < rsi[i] < 60
        
        # Determine target signal based on all filters
        target_signal = 0.0
        if st_flip != 0:
            if st_flip == 1:  # Long flip
                # Must align with 4h uptrend, volume confirmed, RSI in pullback zone
                if trend_4h == 1 and volume_confirmed and rsi_valid_long:
                    target_signal = SIZE
            elif st_flip == -1:  # Short flip
                # Must align with 4h downtrend, volume confirmed, RSI in rally zone
                if trend_4h == -1 and volume_confirmed and rsi_valid_short:
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
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
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
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    profit_target_hit = False
                else:
                    # Position reversal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                    profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals