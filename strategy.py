#!/usr/bin/env python3
"""
Experiment #010: 1h Fisher Transform with 4h/12h HMA Trend and Session Filter

Hypothesis: Previous strategies failed because they used regime-switching (CHOP) which
doesn't work well in crypto's persistent trending nature. This strategy uses:

1. Ehlers Fisher Transform (period=9) - superior to RSI for catching reversals in bear
   markets. Long when Fisher crosses above -1.5, short when crosses below +1.5.
2. 4h HMA(21) for intermediate trend - faster than KAMA, less lag than EMA
3. 12h HMA(50) for major trend bias - align with higher timeframe direction
4. Session filter (8-20 UTC) - only trade during high liquidity hours
5. Volume confirmation - require volume > 1.2x 20-bar average
6. ATR trailing stoploss - 2.5x ATR to protect against reversals

Why this should beat #004 (Sharpe=0.514):
- Fisher Transform catches reversals better than ADX in bear/range markets (2025 test)
- Dual HTF (4h + 12h) provides stronger trend confirmation than single 1d filter
- Session filter reduces false signals during low-liquidity hours
- 1h timeframe with strict filters targets 30-60 trades/year (optimal fee drag)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels (smaller for lower TF)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_12h_hma_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - transforms price into a Gaussian distribution
    for clearer reversal signals. Superior to RSI in bear/range markets.
    Reference: Ehlers, "Cybernetic Analysis for Stocks and Futures"
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate typical price
    typical = (high_s + low_s + close_s) / 3
    
    # Normalize price to -1 to +1 range
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, np.nan)
    
    normalized = 2 * ((typical - lowest) / range_val) - 1
    normalized = normalized.clip(-0.999, 0.999)  # Prevent log domain error
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher_prev = fisher.shift(1)
    
    return fisher.values, fisher_prev.values

def calculate_hma(close, period):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    close_s = pd.Series(close)
    n = len(close)
    
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # Weighted Moving Averages
    wma_half = close_s.rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=False
    )
    wma_full = close_s.rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=False
    )
    
    # Raw HMA calculation
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA on raw HMA
    hma = raw_hma.rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=False
    )
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA(21) for intermediate trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 12h HMA(50) for major trend bias
    hma_12h_50 = calculate_hma(df_12h['close'].values, 50)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 1h indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1h_21 = calculate_hma(close, 21)
    
    # Volume moving average for confirmation
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # UTC hour for session filter
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    # Lower TF = smaller size to reduce fee impact
    BASE_SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        
        if np.isnan(hma_1h_21[i]):
            continue
        
        if np.isnan(volume_ma20[i]) or volume_ma20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high liquidity hours to reduce false signals
        in_session = 8 <= utc_hour[i] <= 20
        
        # === 12H TREND BIAS (Major Direction) ===
        trend_12h_bullish = close[i] > hma_12h_50_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_50_aligned[i]
        
        # === 4H TREND (Intermediate Direction) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 1H HMA SLOPE (Entry Timing) ===
        hma_slope_long = hma_1h_21[i] > hma_1h_21[i-3] if i > 3 else False
        hma_slope_short = hma_1h_21[i] < hma_1h_21[i-3] if i > 3 else False
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_signal = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_signal = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extreme levels for additional confirmation
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * volume_ma20[i]  # 20% above average
        
        # === ENTRY LOGIC (3+ Confluence Required) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need HTF trend + Fisher signal + (session OR volume)
        long_confluence = 0
        if trend_12h_bullish:
            long_confluence += 1  # Major trend alignment
        if trend_4h_bullish:
            long_confluence += 1  # Intermediate trend alignment
        if fisher_long_signal or fisher_oversold:
            long_confluence += 1  # Reversal signal
        if in_session:
            long_confluence += 0.5  # Session filter
        if volume_confirmed:
            long_confluence += 0.5  # Volume confirmation
        if hma_slope_long:
            long_confluence += 0.5  # Momentum confirmation
        
        # Enter long if confluence >= 3.0 (need strong agreement)
        if long_confluence >= 3.0 and trend_12h_bullish and trend_4h_bullish:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Need HTF trend + Fisher signal + (session OR volume)
        short_confluence = 0
        if trend_12h_bearish:
            short_confluence += 1  # Major trend alignment
        if trend_4h_bearish:
            short_confluence += 1  # Intermediate trend alignment
        if fisher_short_signal or fisher_overbought:
            short_confluence += 1  # Reversal signal
        if in_session:
            short_confluence += 0.5  # Session filter
        if volume_confirmed:
            short_confluence += 0.5  # Volume confirmation
        if hma_slope_short:
            short_confluence += 0.5  # Momentum confirmation
        
        # Enter short if confluence >= 3.0 (need strong agreement)
        if short_confluence >= 3.0 and trend_12h_bearish and trend_4h_bearish:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 120 bars (~5 days on 1h), allow weaker entry
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and trend_4h_bullish and fisher_oversold:
                new_signal = BASE_SIZE * 0.6  # Smaller size
            elif trend_12h_bearish and trend_4h_bearish and fisher_overbought:
                new_signal = -BASE_SIZE * 0.6
        
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
            # Exit long if 4h trend turns bearish
            if position_side > 0 and trend_4h_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and trend_4h_bullish:
                trend_reversal = True
        
        # === FISHER REVERSAL EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            # Exit long if Fisher becomes overbought
            if position_side > 0 and fisher[i] > 1.5:
                fisher_exit = True
            # Exit short if Fisher becomes oversold
            if position_side < 0 and fisher[i] < -1.5:
                fisher_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or fisher_exit:
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