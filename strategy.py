#!/usr/bin/env python3
"""
Experiment #006: 12h KAMA-Choppiness Regime with 1d HMA Bias

Hypothesis: After 5 failures, the pattern shows complex multi-condition strategies
fail at higher timeframes. This strategy SIMPLIFIES while keeping proven edges:

1. KAMA (Kaufman Adaptive Moving Average): Adapts to volatility - smooth in trends,
   responsive in ranges. Better than EMA for crypto's regime changes.

2. Choppiness Index regime filter: CHOP>61.8=range (mean revert), CHOP<38.2=trend
   (breakout). This is the KEY differentiator - use right strategy for regime.

3. 1D HMA trend bias: Ultra-stable HTF direction. Only long if price>1d_HMA,
   only short if price<1d_HMA. Prevents counter-trend trades.

4. Asymmetric sizing: Larger positions in trend regime (0.30), smaller in range (0.20)

5. Loose entry thresholds: Ensures ≥10 trades/symbol (learned from failures)

Why this should work where #001-#005 failed:
- Simpler logic = fewer conflicting conditions = more trades
- KAMA adapts to crypto's changing volatility better than fixed EMA
- 1d HMA is more stable than 4h for 12h primary timeframe
- Choppiness filter prevents wrong strategy in wrong regime
- Conservative sizing (0.20-0.30) protects in 2022 crash

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing (wider for 12h)
Target trades: 30-50/year (2-4 per month)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_1d_hma_regime_atr_v1"
timeframe = "12h"
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
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    Smooth in trends, responsive in ranges.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio: |change| / sum of absolute changes
    change = np.abs(close_s - close_s.shift(period))
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    
    # Avoid division by zero
    er = change / volatility.replace(0, np.inf)
    er = er.fillna(0).clip(0, 1)
    
    # Smoothing constant
    sc = (er * (fast / (fast + 1) - slow / (slow + 1)) + slow / (slow + 1)) ** 2
    
    # KAMA calculation
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[period] = close_s.iloc[period]
    
    for i in range(period + 1, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    
    return kama.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True range for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    kama_10 = calculate_kama(close, period=10, fast=2, slow=30)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_TREND = 0.30  # Larger in trend regime
    SIZE_RANGE = 0.20  # Smaller in range regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(kama_10[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 38.2
        is_range_regime = chop_14[i] > 61.8
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND SIGNAL ===
        kama_bull = close[i] > kama_10[i]
        kama_bear = close[i] < kama_10[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === BOLLINGER BAND EXTREMES ===
        near_bb_lower = close[i] < bb_lower[i] * 1.01
        near_bb_upper = close[i] > bb_upper[i] * 0.99
        
        # === POSITION SIZING BY REGIME ===
        current_size = SIZE_TREND if is_trend_regime else SIZE_RANGE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: TREND REGIME - KAMA crossover with HTF bias
        if is_trend_regime:
            # Long: KAMA bull + 1d bullish bias + RSI not overbought
            if kama_bull and bull_bias and not rsi_overbought:
                new_signal = current_size
            
            # Short: KAMA bear + 1d bearish bias + RSI not oversold
            elif kama_bear and bear_bias and not rsi_oversold:
                new_signal = -current_size
        
        # MODE 2: RANGE REGIME - Mean reversion at BB extremes
        elif is_range_regime:
            # Long: Near BB lower + RSI oversold + 1d not strongly bearish
            if near_bb_lower and rsi_oversold:
                if not bear_bias or chop_14[i] > 65:
                    new_signal = current_size
            
            # Short: Near BB upper + RSI overbought + 1d not strongly bullish
            elif near_bb_upper and rsi_overbought:
                if not bull_bias or chop_14[i] > 65:
                    new_signal = -current_size
        
        # MODE 3: TRANSITION REGIME - Use KAMA with looser filters
        else:
            # Long: KAMA bull + 1d not bearish
            if kama_bull and not bear_bias:
                new_signal = current_size
            
            # Short: KAMA bear + 1d not bullish
            elif kama_bear and not bull_bias:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d bias turns bearish strongly
            if position_side > 0 and bear_bias and chop_14[i] < 35:
                trend_reversal = True
            # Exit short if 1d bias turns bullish strongly
            if position_side < 0 and bull_bias and chop_14[i] < 35:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals