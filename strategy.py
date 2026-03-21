#!/usr/bin/env python3
"""
EXPERIMENT #032 - HMA/KAMA Adaptive Trend + Volume + Z-score Filter (30m primary, 4h HTF)
========================================================================================
Hypothesis: 30m timeframe captures intraday trends better than 4h while avoiding 
the noise of 15m. HMA(21) on 4h provides clean trend direction. KAMA(14) on 30m 
adapts to volatility - fast in trends, slow in chop. Volume confirmation ensures 
breakouts have participation. Z-score(20) filter prevents entering at extreme 
overbought/oversold levels (mean reversion risk). This differs from supertrend 
approaches by using adaptive moving averages instead of ATR-based bands.

Key features:
- Primary TF: 30m (this experiment's requirement)
- HTF filter: 4h HMA(21) for major trend direction
- Trend: KAMA(14) on 30m with Efficiency Ratio adaptation
- Entry: Price above/below KAMA + volume > 20-bar average
- Filter: Z-score(20) between -1.5 and +1.5 (avoid extremes)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25 discrete levels (conservative)
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_kama_volume_zscore_30m_4h_v1"
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


def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA)
    KAMA adapts to market noise using Efficiency Ratio (ER)
    ER = |net change| / sum of absolute changes over period
    Higher ER = trending (fast SC), Lower ER = choppy (slow SC)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period - 1, n):
        net_change = abs(close[i] - close[i - period + 1])
        sum_changes = 0.0
        for j in range(i - period + 2, i + 1):
            sum_changes += abs(close[j] - close[j - 1])
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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
    """Calculate rolling Z-score"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Calculate 30m indicators
    kama_30m = calculate_kama(close, period=14, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(kama_30m[i]) or 
            np.isnan(atr[i]) or np.isnan(zscore[i]) or np.isnan(vol_sma[i]) or 
            atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (HTF)
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 30m KAMA trend
        kama_trend = 1 if close[i] > kama_30m[i] else -1
        
        # Volume confirmation (volume > 20-bar average)
        volume_confirmed = volume[i] > vol_sma[i]
        
        # Z-score filter (avoid extremes - mean reversion risk)
        zscore_valid = -1.5 <= zscore[i] <= 1.5
        
        # KAMA slope confirmation (optional - trend momentum)
        kama_slope_valid = True
        if i >= 3:
            kama_slope = kama_30m[i] - kama_30m[i - 3]
            if hma_trend == 1 and kama_slope < 0:
                kama_slope_valid = False
            elif hma_trend == -1 and kama_slope > 0:
                kama_slope_valid = False
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h trend up + 30m above KAMA + volume confirmed + Z-score valid
        if hma_trend == 1 and kama_trend == 1 and volume_confirmed and zscore_valid and kama_slope_valid:
            target_signal = SIZE
        
        # Short entry: 4h trend down + 30m below KAMA + volume confirmed + Z-score valid
        elif hma_trend == -1 and kama_trend == -1 and volume_confirmed and zscore_valid and kama_slope_valid:
            target_signal = -SIZE
        
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
                
                # Check take profit (2R from entry, where R = 2.0*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr[i]:  # 2R = 4*ATR
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
                    if close[i] <= entry_price - 4.0 * atr[i]:  # 2R profit
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
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and kama_trend == -1:
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and kama_trend == 1:
                    # Trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals