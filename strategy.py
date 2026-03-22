#!/usr/bin/env python3
"""
Experiment #003: 1d KAMA Trend + 1w HMA Bias + ADX Regime Filter

Hypothesis: Previous strategies (Choppiness + Connors RSI) failed because they tried to
mean-revert in strongly trending crypto markets. This strategy uses:

1. 1w HMA(21) Major Trend Bias - via mtf_data helper. Only long if price > 1w HMA,
   only short if price < 1w HMA. Prevents counter-trend trades in major moves.

2. 1d KAMA(21) Adaptive Trend - Kaufman's Adaptive MA adjusts smoothing based on ER.
   Fast in trends, slow in chop. More robust than EMA in crypto whipsaws.

3. ADX(14) Regime Filter - ADX > 25 = trending (follow KAMA), ADX < 20 = chop
   (reduce size or stay flat). Prevents entries in low-signal environments.

4. Bollinger Band Squeeze Release - BB Width < 25th percentile (100-bar) then
   price breaks BB. Captures volatility expansion after compression.

5. RSI(14) Momentum Confirmation - RSI > 55 for longs, RSI < 45 for shorts.
   Avoids entering when momentum is fading.

6. ATR(14) Trailing Stop - 2.5x ATR for risk management. Signal → 0 when stopped.

Why this should work on 1d:
- 1d timeframe = 20-50 trades/year target (optimal fee drag)
- 1w HTF filter prevents major counter-trend failures (2022 crash protection)
- KAMA adapts to crypto's variable volatility better than fixed EMA
- ADX filter avoids choppy periods where most strategies fail
- Conservative sizing (0.25-0.35) protects against 77% crashes

Timeframe: 1d (REQUIRED for Experiment #003)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.35 high conviction (1d+1w aligned), 0.15 low conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_bb_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average.
    Adjusts smoothing based on Efficiency Ratio (trend vs noise).
    """
    close_s = pd.Series(close)
    n = period
    
    # Efficiency Ratio: |net change| / sum of absolute changes
    change = np.abs(close_s.diff(period))
    noise = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum()
    er = change / noise.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values (Wilder's smoothing)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    adx = adx.replace([np.inf, -np.inf], np.nan)
    
    return adx.values, plus_di.values, minus_di.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper.values, lower.values, middle.values

def calculate_bb_width(upper, lower, middle):
    """Calculate Bollinger Band Width = (Upper - Lower) / Middle."""
    width = (upper - lower) / middle
    width = pd.Series(width).replace([np.inf, -np.inf], np.nan)
    return width.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate percentile rank of BB Width over lookback period."""
    bb_width_s = pd.Series(bb_width)
    percentile = bb_width_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )
    return percentile.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    kama_21 = calculate_kama(close, period=21)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_middle)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.35
    LOW_CONV_SIZE = 0.15
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(kama_21[i]) or np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_width_pct[i]):
            continue
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === DAILY TREND (KAMA) ===
        kama_bullish = close[i] > kama_21[i]
        kama_bearish = close[i] < kama_21[i]
        
        # KAMA slope (trend direction)
        kama_slope_bull = kama_21[i] > kama_21[i-5] if i >= 5 else False
        kama_slope_bear = kama_21[i] < kama_21[i-5] if i >= 5 else False
        
        # === ADX REGIME FILTER ===
        adx_trending = adx_14[i] > 25
        adx_chop = adx_14[i] < 20
        
        # === BB SQUEEZE DETECTION ===
        bb_squeeze = bb_width_pct[i] < 0.25
        bb_breakout_long = close[i] > bb_upper[i]
        bb_breakout_short = close[i] < bb_lower[i]
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY
        long_score = 0
        
        # Weekly bias (required for high conviction)
        if weekly_bullish:
            long_score += 3
        
        # Daily trend alignment
        if kama_bullish:
            long_score += 2
        if kama_slope_bull:
            long_score += 1
        
        # ADX trend strength
        if adx_trending:
            long_score += 2
        elif not adx_chop:  # neutral ADX
            long_score += 1
        
        # BB squeeze breakout
        if bb_squeeze and bb_breakout_long:
            long_score += 2
        elif bb_breakout_long:
            long_score += 1
        
        # RSI momentum
        if rsi_strong_bull:
            long_score += 1
        elif rsi_bullish:
            long_score += 0.5
        
        # Enter long if score >= 6 (moderate threshold)
        if long_score >= 6:
            if weekly_bullish and kama_bullish and adx_trending:
                new_signal = HIGH_CONV_SIZE  # 0.35 - high conviction
            elif weekly_bullish and kama_bullish:
                new_signal = BASE_SIZE  # 0.25 - base
            else:
                new_signal = LOW_CONV_SIZE  # 0.15 - low conviction
        
        # SHORT ENTRY
        short_score = 0
        
        # Weekly bias (required for high conviction)
        if weekly_bearish:
            short_score += 3
        
        # Daily trend alignment
        if kama_bearish:
            short_score += 2
        if kama_slope_bear:
            short_score += 1
        
        # ADX trend strength
        if adx_trending:
            short_score += 2
        elif not adx_chop:  # neutral ADX
            short_score += 1
        
        # BB squeeze breakout
        if bb_squeeze and bb_breakout_short:
            short_score += 2
        elif bb_breakout_short:
            short_score += 1
        
        # RSI momentum
        if rsi_strong_bear:
            short_score += 1
        elif rsi_bearish:
            short_score += 0.5
        
        # Enter short if score >= 6
        if short_score >= 6:
            if weekly_bearish and kama_bearish and adx_trending:
                new_signal = -HIGH_CONV_SIZE  # -0.35 - high conviction
            elif weekly_bearish and kama_bearish:
                new_signal = -BASE_SIZE  # -0.25 - base
            else:
                new_signal = -LOW_CONV_SIZE  # -0.15 - low conviction
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~30 days on 1d), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if weekly_bullish and kama_bullish and rsi_bullish:
                new_signal = LOW_CONV_SIZE
            elif weekly_bearish and kama_bearish and rsi_bearish:
                new_signal = -LOW_CONV_SIZE
        
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
            # Exit long if weekly trend turns bearish
            if position_side > 0 and weekly_bearish:
                trend_reversal = True
            # Exit short if weekly trend turns bullish
            if position_side < 0 and weekly_bullish:
                trend_reversal = True
        
        # === KAMA CROSSOVER EXIT ===
        kama_exit = False
        if in_position and position_side != 0:
            # Exit long if price crosses below KAMA
            if position_side > 0 and close[i] < kama_21[i]:
                kama_exit = True
            # Exit short if price crosses above KAMA
            if position_side < 0 and close[i] > kama_21[i]:
                kama_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or kama_exit:
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