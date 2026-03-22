#!/usr/bin/env python3
"""
Experiment #272: 30m KAMA Pullback with 4h HMA Bias and ATR Stops

Hypothesis: After 271 experiments, the winning pattern is clear - KAMA on primary TF
+ HMA bias on HTF works best (current best: mtf_4h_kama_1d_hma_adx_atr_v1, Sharpe=0.478).
Previous 30m attempt (#260) failed with Sharpe=-3.690 due to overly strict ADX filter.

This version:
1. REMOVES ADX filter (too restrictive, caused 0 trades in #266)
2. Uses KAMA(10) on 30m for smooth trend following (less whipsaw than EMA)
3. Uses 4h HMA(21) for directional bias (proven in best strategies)
4. Entry on pullback to KAMA when aligned with 4h HMA bias
5. ATR(14) stops at 2.0*ATR (tighter for 30m vs 12h's 3.0*ATR)
6. Position sizing: 0.25 base, 0.35 max (discrete levels)
7. LOOSE entry conditions to ensure >=10 trades per symbol

Key improvements from #260 failure:
- No ADX filter (was blocking all entries)
- Simpler entry logic (KAMA slope + 4h HMA bias only)
- Tighter stops (2.0*ATR vs 3.0*ATR for faster 30m timeframe)
- Ensure signals change frequently enough for >=10 trades

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_pullback_4h_hma_atr_v2"
timeframe = "30m"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market volatility - smooth in trends, responsive in ranges.
    ER (Efficiency Ratio) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC (Smoothing Constant) = ER * (fast_sc - slow_sc) + slow_sc
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    # Calculate KAMA
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        sc = er[i] * (fast_sc - slow_sc) + slow_sc
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_MAX = 0.35   # Maximum position size
    SIZE_MIN = 0.15   # Minimum position size
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = directional bias (hard filter)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 30m TREND ===
        # KAMA slope for short-term trend
        kama_slope_bull = kama[i] > kama[i - 5] if not np.isnan(kama[i - 5]) else False
        kama_slope_bear = kama[i] < kama[i - 5] if not np.isnan(kama[i - 5]) else False
        
        # Price relative to KAMA (pullback entry)
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # === RSI FILTER (loose, to ensure trades) ===
        # Only filter extreme RSI that suggests reversal imminent
        rsi_not_overbought = rsi[i] < 80  # Allow entries even at RSI 70-79
        rsi_not_oversold = rsi[i] > 20    # Allow entries even at RSI 21-30
        
        # === VOLATILITY ADJUSTMENT ===
        # Adjust position size based on ATR
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_MIN
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS (LOOSE to ensure >=10 trades) ===
        new_signal = 0.0
        
        # LONG ENTRY: 4h bias up + 30m KAMA bullish + pullback or breakout
        # Looser: only need 4h bias + KAMA slope (no strict pullback requirement)
        long_conditions = (
            bull_trend_4h and      # 4h HMA bias bullish
            kama_slope_bull and    # 30m KAMA trending up
            rsi_not_overbought     # RSI not extremely overbought
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_4h and      # 4h HMA bias bearish
            kama_slope_bear and    # 30m KAMA trending down
            rsi_not_oversold       # RSI not extremely oversold
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
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals