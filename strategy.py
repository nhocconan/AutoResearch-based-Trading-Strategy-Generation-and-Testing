#!/usr/bin/env python3
"""
EXPERIMENT #025 - HMA Trend + RSI Momentum + Volume Confirmation (15m primary, 1h HTF)
======================================================================================
Hypothesis: 15m HMA(21) captures fast trend changes better than EMA/KAMA on short TF.
RSI(14) momentum filter ensures we enter on pullbacks (not chasing). Volume ratio
confirms genuine moves vs fakeouts. 1h HMA(50) provides HTF trend alignment without
being too slow (4h was too laggy for 15m entries in previous attempts).

Key differences from failed strategies:
- Uses HMA (faster response than KAMA/EMA) instead of Supertrend
- 1h HTF (not 4h) for better alignment with 15m entries
- Volume ratio filter (not just raw volume) to confirm breakouts
- Z-score on price for mean reversion filter (avoid extreme entries)

Risk management:
- Position size: 0.25-0.30 discrete levels
- Stoploss: 2.0*ATR(14) trailing
- Take profit: Reduce to half at 2R, trail stop at 1R
- Max signal magnitude: 0.35
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_volume_zscore_15m_1h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average - faster response than EMA"""
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


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values


def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope direction"""
    n = len(hma_values)
    slope = np.zeros(n)
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i - lookback]):
            slope[i] = 1 if hma_values[i] > hma_values[i - lookback] else -1
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    hma_1h = calculate_hma(df_1h['close'].values, 50)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)  # auto shift(1)
    
    # Calculate 15m indicators (pre-compute before loop for performance)
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 9)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    hma_slope = calculate_hma_slope(hma_15m, 5)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    MAX_SIZE = 0.35  # Maximum position size
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 1.0  # ATR at entry for R calculation
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1h_aligned[i]) or np.isnan(hma_15m[i]) or 
            np.isnan(hma_15m_fast[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or 
            np.isnan(zscore[i]) or np.isnan(vol_ratio[i]) or np.isnan(hma_slope[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # HTF trend filter (1h HMA50)
        htf_trend = 1 if close[i] > hma_1h_aligned[i] else -1
        
        # 15m trend filter (HMA21 slope)
        ltf_trend = int(hma_slope[i])
        
        # HMA crossover signal (fast vs slow)
        hma_cross = 1 if hma_15m_fast[i] > hma_15m[i] else -1
        
        # RSI momentum filter (avoid extremes, prefer pullbacks)
        rsi_valid_long = 35 <= rsi[i] <= 65  # Not overbought
        rsi_valid_short = 35 <= rsi[i] <= 65  # Not oversold
        rsi_momentum_long = rsi[i] > 50  # Bullish momentum
        rsi_momentum_short = rsi[i] < 50  # Bearish momentum
        
        # Z-score filter (avoid extreme mean reversion entries)
        zscore_valid = abs(zscore[i]) < 2.0  # Not at extreme
        
        # Volume confirmation (volume > 1.2x average for conviction)
        volume_confirmed = vol_ratio[i] > 1.0
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HTF bullish + LTF bullish + HMA cross bullish + RSI valid + Z-score ok + Volume
        if (htf_trend == 1 and ltf_trend == 1 and hma_cross == 1 and 
            rsi_valid_long and rsi_momentum_long and zscore_valid and volume_confirmed):
            target_signal = SIZE
        
        # Short entry: HTF bearish + LTF bearish + HMA cross bearish + RSI valid + Z-score ok + Volume
        elif (htf_trend == -1 and ltf_trend == -1 and hma_cross == -1 and 
              rsi_valid_short and rsi_momentum_short and zscore_valid and volume_confirmed):
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            r_multiple = 2.0 * entry_atr  # R = 2*ATR
            
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 2 * r_multiple:  # 2R = 4*ATR from entry
                        take_profit_triggered = True
                
                # Check trend reversal (HMA cross bearish)
                if hma_cross == -1:
                    trend_reversal = True
                    
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    if close[i] <= entry_price - 2 * r_multiple:  # 2R profit
                        take_profit_triggered = True
                
                # Check trend reversal (HMA cross bullish)
                if hma_cross == 1:
                    trend_reversal = True
        
        # Apply signal based on conditions
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
            entry_atr = 1.0
            
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            
        elif trend_reversal and position_side != 0:
            # Trend reversed, exit position
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
            entry_atr = 1.0
            
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
                # Maintain existing position
                if not profit_target_hit:
                    signals[i] = SIZE * position_side
                else:
                    signals[i] = HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals