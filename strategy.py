#!/usr/bin/env python3
"""
EXPERIMENT #051 - Supertrend + RSI Pullback + 4h HMA Trend Filter (1h primary)
=====================================================================================
Hypothesis: 1h Supertrend captures trend moves faster than 12h Donchian, but needs
strong HTF filter to avoid chop. 4h HMA(50) provides proven trend direction filter.
RSI(14) pullback entries (RSI<45 in uptrend, RSI>55 in downtrend) improve entry timing
vs pure breakout. Volume confirmation filters false signals. This differs from #047
by using Supertrend (more responsive) + RSI pullback (better timing) on 1h vs 12h Donchian.

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filter: 4h HMA(50) for trend direction (proven in successful strategies)
- Trend: Supertrend(10, 3.0) for entry signals
- Entry: Supertrend flip + RSI pullback confirmation (RSI<45 long, RSI>55 short)
- Volume filter: volume > 1.2 * SMA(volume, 20) for confirmation
- Regime: 4h HMA slope > 0 for long bias, < 0 for short bias
- Stoploss: 2.0*ATR(14) trailing stop
- Position sizing: 0.25-0.30 discrete, scaled by Supertrend strength
- Take profit: Reduce to half at 2R profit, trail stop

Why this should beat #047 (Sharpe=0.490):
- 1h Supertrend catches moves earlier than 12h Donchian
- RSI pullback = better entry timing (buy dips, not breakouts)
- 4h HMA filter = same proven HTF approach as current best
- More trades than 12h strategy while maintaining quality filters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_pullback_1h_4h_v1"
timeframe = "1h"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    
    # Final upper and lower bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)  # 1 = uptrend, -1 = downtrend
    
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    
    for i in range(1, n):
        # Final Upper Band
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]
        
        # Final Lower Band
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]
        
        # Supertrend and trend direction
        if trend[i - 1] == 1:
            if close[i] < final_lower[i]:
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                trend[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                trend[i] = -1
                supertrend[i] = final_upper[i]
    
    return supertrend, trend, final_upper, final_lower


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period - 1, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume"""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    supertrend, st_trend, st_upper, st_lower = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Calculate HMA slope for regime filter (compare current vs 5 bars ago)
    hma_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i - 5]):
            hma_slope[i] = hma_4h_aligned[i] - hma_4h_aligned[i - 5]
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.32   # Max position size
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(supertrend[i]) or
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma[i]) or
            np.isnan(hma_slope[i]) or atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_bullish = price_above_4h_hma and hma_slope[i] > 0
        hma_bearish = not price_above_4h_hma and hma_slope[i] < 0
        
        # Supertrend signals
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Supertrend flip detection (entry signal)
        st_flip_long = (i > 0 and st_trend[i] == 1 and st_trend[i - 1] == -1)
        st_flip_short = (i > 0 and st_trend[i] == -1 and st_trend[i - 1] == 1)
        
        # RSI pullback filter (buy dips in uptrend, sell rallies in downtrend)
        rsi_pullback_long = rsi[i] < 45  # RSI pulled back in uptrend
        rsi_pullback_short = rsi[i] > 55  # RSI rallied in downtrend
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # Calculate position size based on trend strength
        trend_strength = abs(hma_slope[i]) / (hma_4h_aligned[i] * 0.01 + 1e-10)
        size_multiplier = min(1.0 + trend_strength * 0.1, 1.15)
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * size_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend flip + 4h HMA bullish + RSI pullback + volume
        if (st_flip_long and hma_bullish and rsi_pullback_long and volume_confirmed):
            target_signal = position_size
        
        # Short entry: Supertrend flip + 4h HMA bearish + RSI pullback + volume
        elif (st_flip_short and hma_bearish and rsi_pullback_short and volume_confirmed):
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
                # Exit if Supertrend flips against position OR 4h HMA alignment breaks
                st_reversal_long = st_trend[i] == -1
                st_reversal_short = st_trend[i] == 1
                hma_alignment_broken = (position_side == 1 and not hma_bullish) or \
                                       (position_side == -1 and not hma_bearish)
                
                if st_reversal_long or st_reversal_short or hma_alignment_broken:
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