#!/usr/bin/env python3
"""
EXPERIMENT #013 - HMA Crossover + Z-Score + Volume Filter (15m primary, 1h/4h HTF)
==================================================================================
Hypothesis: 15m HMA(8/21) crossover provides fast entry signals, but needs filters
to avoid whipsaws. 1h HMA(34) determines major trend direction. 4h HMA(50) provides
regime filter. Z-score(20) avoids entering at price extremes (>2.0 std). Volume
spike confirmation (>1.5x 20-bar avg) ensures real moves not fakeouts.

Why 15m: Faster than 1h/4h strategies, captures intraday swings. Previous 15m
strategies failed due to no HTF filter or poor position sizing. This combines:
- Fast 15m HMA crossover for entries
- 1h HMA trend filter (not daily - too slow for 15m)
- 4h HMA regime filter (avoid counter-trend trades)
- Z-score mean reversion filter (don't chase extremes)
- Volume confirmation (real moves have volume)
- Tight 2.0*ATR stoploss (15m moves fast)

Key differences from failed #007 supertrend_adx_rsi_15m_4h:
- HMA instead of Supertrend (faster, less lag)
- Z-score filter instead of ADX (avoids extremes)
- Volume confirmation (missing in #007)
- Smaller position size (0.25 vs implied 1.0)
- Proper MTF alignment with align_htf_to_ltf()
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_zscore_volume_15m_1h_4h_v1"
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 34)
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - uses shift(1) internally)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    atr = calculate_atr(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative for 15m)
    HALF_SIZE = SIZE / 2
    
    # Track position state
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    profit_target_hit = False
    
    min_period = 100  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or
            np.isnan(atr[i]) or np.isnan(zscore[i]) or np.isnan(vol_ratio[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # HTF trend filters
        h1h_trend = 1 if close[i] > hma_1h_aligned[i] else -1
        h4h_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 15m HMA crossover signal
        hma_cross = 0
        if hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]:
            hma_cross = 1  # Bullish crossover
        elif hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]:
            hma_cross = -1  # Bearish crossover
        
        # Z-score filter (avoid extremes > 2.0 std)
        zscore_valid_long = zscore[i] < 1.5  # Don't buy at extreme highs
        zscore_valid_short = zscore[i] > -1.5  # Don't sell at extreme lows
        
        # Volume confirmation (real moves have volume)
        volume_confirmed = vol_ratio[i] > 1.3  # 30% above average
        
        # Determine target signal
        target_signal = 0.0
        
        # Long: HMA cross up + 1h trend up + 4h trend up + Z-score ok + volume
        if (hma_cross == 1 and h1h_trend == 1 and h4h_trend == 1 and 
            zscore_valid_long and volume_confirmed):
            target_signal = SIZE
        
        # Short: HMA cross down + 1h trend down + 4h trend down + Z-score ok + volume
        elif (hma_cross == -1 and h1h_trend == -1 and h4h_trend == -1 and 
              zscore_valid_short and volume_confirmed):
            target_signal = -SIZE
        
        # Stoploss and take profit logic
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * atr[i]:
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            profit_target_hit = False
        elif take_profit_triggered:
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            if target_signal != 0.0 and position_side == 0:
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                entry_price = close[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Exit if HTF trend reverses against position
                if position_side == 1 and h1h_trend == -1:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    profit_target_hit = False
                elif position_side == -1 and h1h_trend == 1:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    profit_target_hit = False
                else:
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals