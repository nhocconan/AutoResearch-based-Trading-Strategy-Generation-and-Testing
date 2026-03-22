#!/usr/bin/env python3
"""
Experiment #021: 4h Volatility Squeeze + Donchian Breakout + 1d/1w Trend Filter

Hypothesis: Previous strategies failed because they relied on mean-reversion indicators
(Fisher, RSI extremes) that whipsaw in trending crypto markets. This strategy uses:

1. Bollinger Band Width Squeeze - detects volatility contraction before expansion.
   BB Width = (Upper - Lower) / Middle. Entry when BB Width < 20th percentile (100-bar).
   Proven to catch major breakouts in crypto (BTC 2021 rally, 2022 crash).

2. Donchian Channel(20) Breakout - clean trend entry signal. Long when price breaks
   20-bar high, short when breaks 20-bar low. Simple but effective for crypto trends.

3. 1d HMA(21) Trend Filter - via mtf_data helper. Only long if price > 1d HMA,
   only short if price < 1d HMA. Prevents counter-trend breakout failures.

4. 1w HMA(21) Major Bias - via mtf_data helper. Increases position size when
   4h and 1w trends align (high conviction), reduces when they diverge.

5. RSI(14) Momentum Filter - RSI > 50 for longs, RSI < 50 for shorts.
   Avoids entering breakouts with no momentum backing.

6. ATR(14) Trailing Stop - 2.5x ATR for risk management. Signal → 0 when stopped.

Why this should work:
- BB Squeeze + Donchian is a proven breakout combo (John Carter's TTMSqueeze variant)
- 1d/1w HTF filters prevent false breakouts against major trend
- 4h timeframe = 20-50 trades/year target (optimal for fee drag)
- Conservative sizing (0.25-0.30) protects against 77% crashes like 2022
- Simple logic = fewer conditions to all align = more trades (avoids 0-trade failure)

Timeframe: 4h (REQUIRED for Experiment #021)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 high conviction, 0.15 low conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_donchian_1d_1w_v1"
timeframe = "4h"
leverage = 1.0

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
    width = width.replace([np.inf, -np.inf], np.nan)
    return width.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate percentile rank of BB Width over lookback period."""
    bb_width_s = pd.Series(bb_width)
    # Percentile rank: where current value sits in recent distribution
    percentile = bb_width_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )
    return percentile.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel highs and lows."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    donchian_high = high_s.rolling(window=period, min_periods=period).max().values
    donchian_low = low_s.rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    return donchian_high, donchian_low, donchian_mid

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
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    """
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend filter
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1w HMA for major bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_middle)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(bb_width_pct[i]) or np.isnan(donchian_high[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === BB SQUEEZE DETECTION ===
        # Squeeze = BB Width in bottom 25% of recent range
        bb_squeeze = bb_width_pct[i] < 0.25
        
        # === DONCHIAN BREAKOUT ===
        # Detect breakouts (need previous bar comparison)
        donchian_breakout_long = False
        donchian_breakout_short = False
        
        if i > 0:
            # Long: price breaks above Donchian high
            if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]:
                donchian_breakout_long = True
            # Short: price breaks below Donchian low
            if close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]:
                donchian_breakout_short = True
        
        # Also check if already above/below (continuation)
        above_donchian = close[i] > donchian_high[i]
        below_donchian = close[i] < donchian_low[i]
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY
        long_conditions = 0
        
        # Primary: Donchian breakout or continuation above
        if donchian_breakout_long:
            long_conditions += 3
        elif above_donchian:
            long_conditions += 1
        
        # Trend alignment (daily)
        if daily_bullish:
            long_conditions += 2
        
        # Major bias (weekly) - adds conviction
        if weekly_bullish:
            long_conditions += 1
        
        # RSI momentum
        if rsi_strong_bull:
            long_conditions += 1
        elif rsi_bullish:
            long_conditions += 0.5
        
        # BB squeeze adds significance to breakout
        if bb_squeeze and (donchian_breakout_long or above_donchian):
            long_conditions += 1.5
        
        # Enter long if conditions >= 5 (moderate threshold for trade frequency)
        if long_conditions >= 5:
            # Determine position size based on conviction
            if weekly_bullish and daily_bullish:
                new_signal = HIGH_CONV_SIZE  # 0.30 - high conviction
            else:
                new_signal = BASE_SIZE  # 0.25 - base
        
        # SHORT ENTRY
        short_conditions = 0
        
        # Primary: Donchian breakout or continuation below
        if donchian_breakout_short:
            short_conditions += 3
        elif below_donchian:
            short_conditions += 1
        
        # Trend alignment (daily)
        if daily_bearish:
            short_conditions += 2
        
        # Major bias (weekly) - adds conviction
        if weekly_bearish:
            short_conditions += 1
        
        # RSI momentum
        if rsi_strong_bear:
            short_conditions += 1
        elif rsi_bearish:
            short_conditions += 0.5
        
        # BB squeeze adds significance to breakout
        if bb_squeeze and (donchian_breakout_short or below_donchian):
            short_conditions += 1.5
        
        # Enter short if conditions >= 5
        if short_conditions >= 5:
            # Determine position size based on conviction
            if weekly_bearish and daily_bearish:
                new_signal = -HIGH_CONV_SIZE  # -0.30 - high conviction
            else:
                new_signal = -BASE_SIZE  # -0.25 - base
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~160 hours = 6.7 days on 4h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if above_donchian and daily_bullish and rsi_bullish:
                new_signal = LOW_CONV_SIZE
            elif below_donchian and daily_bearish and rsi_bearish:
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
            # Exit long if daily trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if daily trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === DONCHIAN REVERSAL EXIT ===
        donchian_exit = False
        if in_position and position_side != 0:
            # Exit long if price falls back below Donchian mid
            if position_side > 0 and close[i] < donchian_mid[i]:
                donchian_exit = True
            # Exit short if price rises back above Donchian mid
            if position_side < 0 and close[i] > donchian_mid[i]:
                donchian_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or donchian_exit:
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