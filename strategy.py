#!/usr/bin/env python3
"""
Experiment #014: 4h KAMA Adaptive Trend with Fisher Transform Entries

Hypothesis: Previous strategies failed due to over-filtering (0 trades) or using
indicators that don't adapt to regime changes. This strategy uses:

1. KAMA (Kaufman Adaptive Moving Average) - adapts speed based on market efficiency
   - Fast in trends, slow in chop - automatically adjusts without regime detection
2. Fisher Transform - normalized oscillator that catches reversals cleanly
   - Entry when Fisher crosses -1.5 (long) or +1.5 (short)
3. 12h KAMA for intermediate trend bias
4. 1d KAMA for major trend filter
5. ATR(14) stoploss at 2.5x - protects against crashes
6. Simple entry logic - fewer filters = more trades (avoid 0-trade failure)

Why this should work:
- KAMA outperforms EMA/HMA in mixed regime markets (2021-2024 had both)
- Fisher Transform has proven reversal capture in bear markets
- 4h TF targets 25-45 trades/year (optimal for fee/return balance)
- Simpler logic than failed CRSI strategies = more trade generation
- Adaptive nature handles 2022 crash + 2025 bear without manual regime switch

Timeframe: 4h (REQUIRED)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_12h_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in choppy markets.
    
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    price_change = np.abs(close_s - close_s.shift(period))
    volatility = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    
    er = price_change / volatility
    er = er.fillna(0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for cleaner reversal signals.
    
    Price = (0.67 * (close - low_n) / (high_n - low_n) - 0.67) + 0.67 * prev
    Fisher = 0.5 * ln((1 + Price) / (1 - Price))
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    n = len(close)
    
    # Normalize price
    hl_range = high_s.rolling(window=period, min_periods=period).max() - \
               low_s.rolling(window=period, min_periods=period).min()
    
    price_norm = np.zeros(n)
    price_norm[0] = 0.0
    
    for i in range(period, n):
        if hl_range.iloc[i] > 0:
            raw = 0.67 * ((close[i] - low_s.rolling(window=period, min_periods=period).min().iloc[i]) / 
                         hl_range.iloc[i] - 0.5) + 0.67 * price_norm[i-1]
            price_norm[i] = np.clip(raw, -0.999, 0.999)
        else:
            price_norm[i] = price_norm[i-1]
    
    # Fisher Transform
    fisher = np.zeros(n)
    for i in range(period, n):
        if abs(price_norm[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1 + price_norm[i]) / (1 - price_norm[i]))
        else:
            fisher[i] = fisher[i-1]
    
    return fisher

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return atr.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h KAMA for intermediate trend
    kama_12h_20 = calculate_kama(df_12h['close'].values, period=20)
    kama_12h_20_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_20)
    
    # Calculate 1d KAMA for major trend bias
    kama_1d_20 = calculate_kama(df_1d['close'].values, period=20)
    kama_1d_20_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_20)
    
    # Calculate 4h indicators
    kama_4h_14 = calculate_kama(close, period=14)
    kama_4h_30 = calculate_kama(close, period=30)
    fisher = calculate_fisher_transform(high, low, close, period=9)
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_12h_20_aligned[i]) or np.isnan(kama_1d_20_aligned[i]):
            continue
        
        if np.isnan(kama_4h_14[i]) or np.isnan(kama_4h_30[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > kama_1d_20_aligned[i]
        daily_bearish = close[i] < kama_1d_20_aligned[i]
        
        # === 12H INTERMEDIATE TREND ===
        hma_12h_bullish = close[i] > kama_12h_20_aligned[i]
        hma_12h_bearish = close[i] < kama_12h_20_aligned[i]
        
        # === 4H SHORT-TERM TREND ===
        kama_4h_bullish = kama_4h_14[i] > kama_4h_30[i]
        kama_4h_bearish = kama_4h_14[i] < kama_4h_30[i]
        
        # === TREND STRENGTH ===
        trend_strong = adx_14[i] > 25
        trend_weak = adx_14[i] < 20
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > -1.5 and fisher[i-1] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher[i-1] >= 1.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Fisher oversold + trend alignment
        long_score = 0
        
        # Primary trigger: Fisher cross or extreme
        if fisher_cross_up or fisher_oversold:
            long_score += 2.5
        
        # Trend alignment (need at least 2 of 3 timeframes bullish)
        trend_bullish_count = sum([daily_bullish, hma_12h_bullish, kama_4h_bullish])
        if trend_bullish_count >= 2:
            long_score += 2.0
        elif trend_bullish_count >= 1:
            long_score += 1.0
        
        # ADX filter - prefer trending but allow weak trend for mean reversion
        if trend_strong and kama_4h_bullish:
            long_score += 1.0
        elif trend_weak:
            long_score += 0.5
        
        # Enter long if score >= 4.5 (moderate confluence - not too strict)
        if long_score >= 4.5:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Fisher overbought + trend alignment
        short_score = 0
        
        # Primary trigger: Fisher cross or extreme
        if fisher_cross_down or fisher_overbought:
            short_score += 2.5
        
        # Trend alignment (need at least 2 of 3 timeframes bearish)
        trend_bearish_count = sum([daily_bearish, hma_12h_bearish, kama_4h_bearish])
        if trend_bearish_count >= 2:
            short_score += 2.0
        elif trend_bearish_count >= 1:
            short_score += 1.0
        
        # ADX filter
        if trend_strong and kama_4h_bearish:
            short_score += 1.0
        elif trend_weak:
            short_score += 0.5
        
        # Enter short if score >= 4.5
        if short_score >= 4.5:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~10 days on 4h), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if fisher_oversold and trend_bullish_count >= 1:
                new_signal = REDUCED_SIZE
            elif fisher_overbought and trend_bearish_count >= 1:
                new_signal = -REDUCED_SIZE
        
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
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long if Fisher goes overbought
            if position_side > 0 and fisher[i] > 1.0:
                fisher_exit = True
            # Exit short if Fisher goes oversold
            if position_side < 0 and fisher[i] < -1.0:
                fisher_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if major trend turns bearish (both 12h and 1d)
            if position_side > 0 and hma_12h_bearish and daily_bearish:
                trend_reversal = True
            # Exit short if major trend turns bullish
            if position_side < 0 and hma_12h_bullish and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or fisher_exit or trend_reversal:
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals