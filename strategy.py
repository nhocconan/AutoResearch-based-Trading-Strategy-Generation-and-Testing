#!/usr/bin/env python3
"""
Experiment #309: 1h KAMA Crossover with Dual HTF HMA Bias and ATR Stop

Hypothesis: After analyzing 298+ experiments, clear patterns emerge for 1h timeframe:
1. Supertrend on 1h FAILED twice (#297 Sharpe=-1.585, #303 Sharpe=-0.891)
2. Complex ensembles with 5+ filters generate 0 trades (#307, #308)
3. Donchian breakout showed promise on 12h (#299 Return=+33%) but needs faster TF
4. KAMA (Kaufman Adaptive MA) adapts to volatility better than EMA/HMA in choppy 1h markets
5. Dual HTF bias (4h + 1d HMA) proven edge from #292/#304 (Sharpe=0.485)

This strategy uses KAMA CROSSOVER (adaptive trend following):
1. 1d HMA(21) for primary directional bias (proven edge)
2. 4h HMA(21) for intermediate trend confirmation
3. KAMA(10,2,30) crossover on 1h for entry timing (adaptive to volatility)
4. ATR(14) trailing stoploss at 2.0x (tighter for 1h vs 12h's 2.5x)
5. Simple entry: KAMA fast crosses slow + HTF bias aligned

Why this might work on 1h:
- KAMA adapts ER (Efficiency Ratio) - slow in chop, fast in trends
- 1h needs faster response than 12h Donchian but slower than 15m EMA
- Dual HTF filter reduces false signals without over-filtering
- Simpler than #297's 5-indicator ensemble (that failed badly)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_crossover_dual_htf_hma_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |change| / sum(|individual changes|)
    SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    
    Parameters from Kaufman's original work:
    - fast_period=2 (fast SC = 2/(2+1) = 0.6667)
    - slow_period=30 (slow SC = 2/(30+1) = 0.0645)
    - er_period=10 (lookback for ER calculation)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio components
    change = np.abs(close - np.roll(close, er_period))
    change[0:er_period] = np.nan
    
    # Sum of absolute individual changes
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1))[1:])
    
    volatility[0:er_period] = np.nan
    
    # Efficiency Ratio
    er = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing Constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Dynamic Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=20, fast_period=5, slow_period=50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_INCREASED = 0.30  # Increased size in strong trend
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (Dual HTF Filter) ===
        # 4h HMA = intermediate trend
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1d HMA = primary directional bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA CROSSOVER ===
        # Fast KAMA crosses above Slow KAMA = bullish
        kama_bullish = kama_fast[i] > kama_slow[i]
        # Fast KAMA crosses below Slow KAMA = bearish
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # Check for actual crossover (not just above/below)
        kama_cross_bull = False
        kama_cross_bear = False
        
        if i > 0 and not np.isnan(kama_fast[i-1]) and not np.isnan(kama_slow[i-1]):
            kama_cross_bull = (kama_fast[i-1] <= kama_slow[i-1]) and (kama_fast[i] > kama_slow[i])
            kama_cross_bear = (kama_fast[i-1] >= kama_slow[i-1]) and (kama_fast[i] < kama_slow[i])
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_BASE  # Conservative in high vol
        else:
            position_size = SIZE_INCREASED
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 4h bias up + 1d bias up + KAMA crossover
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_4h and  # 4h HMA bias bullish
            bull_trend_1d and  # 1d HMA bias bullish
            kama_cross_bull  # KAMA crossover signal
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_4h and  # 4h HMA bias bearish
            bear_trend_1d and  # 1d HMA bias bearish
            kama_cross_bear  # KAMA crossover signal
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and (bear_trend_4h or bear_trend_1d):
                new_signal = 0.0  # HTF trend reversed against long
            if position_side < 0 and (bull_trend_4h or bull_trend_1d):
                new_signal = 0.0  # HTF trend reversed against short
        
        # === KAMA REVERSAL EXIT ===
        # Exit if KAMA crossover reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and kama_cross_bear:
                new_signal = 0.0  # KAMA crossed against long
            if position_side < 0 and kama_cross_bull:
                new_signal = 0.0  # KAMA crossed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals