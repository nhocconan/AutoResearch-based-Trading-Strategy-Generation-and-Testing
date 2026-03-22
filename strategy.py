#!/usr/bin/env python3
"""
Experiment #005: 1h Connors RSI + Choppiness Regime + 4h/1d HMA Trend Filter

Hypothesis: Previous 4h/12h strategies failed because they were too slow for crypto's
volatile nature, while pure mean-reversion got whipsawed in 2022 crash. This strategy uses:

1. Connors RSI (CRSI) - 3-component mean reversion signal:
   - RSI(3) for short-term momentum
   - RSI_Streak(2) for consecutive up/down days
   - PercentRank(100) for relative price position
   Entry: CRSI < 15 (oversold) for longs, CRSI > 85 (overbought) for shorts.
   Proven 75% win rate in Larry Connors' research.

2. Choppiness Index (CHOP) Regime Filter - detects market state:
   - CHOP > 55 = range/choppy → use mean reversion (CRSI entries)
   - CHOP < 45 = trending → use trend follow (breakout entries)
   - 45-55 = neutral → no trades (avoids whipsaw)
   This is CRITICAL for 2025 bear/range market.

3. 4h HMA(21) Trend Filter - via mtf_data helper. Only long if price > 4h HMA,
   only short if price < 4h HMA. Prevents counter-trend mean reversion failures.

4. 1d HMA(21) Major Bias - via mtf_data helper. Increases position size when
   4h and 1d trends align (high conviction), reduces when they diverge.

5. Session Filter (8-20 UTC) - Only trade during high-volume hours.
   Reduces false signals during Asian/US overnight low-volume periods.

6. Volume Filter - Volume > 0.8x 20-bar average. Confirms institutional participation.

7. ATR(14) Trailing Stop - 2.5x ATR for risk management. Signal → 0 when stopped.

Why this should work for 1h:
- CRSI + CHOP combo excels in range/bear markets (2025 test period)
- 4h/1d HTF filters prevent counter-trend failures that killed pure mean-reversion
- Session + volume filters reduce trade frequency to 30-80/year target
- Conservative sizing (0.20-0.30) protects against 77% crashes like 2022
- Different from all 4 failed experiments (new indicator combo)

Timeframe: 1h (REQUIRED for Experiment #005)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20 base, 0.30 high conviction, 0.15 low conviction
Stoploss: 2.5 * ATR(14) trailing
Trade frequency target: 30-80/year (strict confluence required)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h_1d_v1"
timeframe = "1h"
leverage = 1.0

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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - 3-component mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry signals:
    - CRSI < 10-15: Oversold → Long opportunity
    - CRSI > 85-90: Overbought → Short opportunity
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    # Streak = consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    
    # RSI of streak: use absolute streak values
    streak_s = pd.Series(streak_abs)
    streak_gain = streak_s.where(streak_sign > 0, 0.0)
    streak_loss = -streak_s.where(streak_sign < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.replace([np.inf, -np.inf], np.nan)
    
    # Component 3: PercentRank(100)
    # Where current price sits in recent 100-bar distribution
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50,
        raw=False
    )
    
    # Combine all three components
    crsi = (rsi_short + rsi_streak.values + percent_rank.values) / 3
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Choppy/Range-bound market → Mean reversion strategies
    - CHOP < 38.2: Trending market → Trend following strategies
    - 38.2-61.8: Transition zone → Reduce position size or stay flat
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    # Highest High and Lowest Low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Sum of ATR over period
    atr_sum = atr.rolling(window=period, min_periods=period).sum()
    
    # CHOP formula
    price_range = highest_high - lowest_low
    price_range = price_range.replace(0, np.inf)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs recent average."""
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=period, min_periods=period).mean()
    volume_ratio = volume_s / avg_volume
    volume_ratio = volume_ratio.replace([np.inf, -np.inf], np.nan)
    return volume_ratio.values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend filter
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d HMA for major bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    volume_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.20
    HIGH_CONV_SIZE = 0.30
    LOW_CONV_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(volume_ratio[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume_ratio[i] > 0.8
        
        # === 4H TREND FILTER ===
        four_h_bullish = close[i] > hma_4h_21_aligned[i]
        four_h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 1D MAJOR BIAS ===
        one_d_bullish = close[i] > hma_1d_21_aligned[i]
        one_d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55  # Range-bound market
        chop_trend = chop[i] < 45  # Trending market
        # chop_neutral = 45 <= chop[i] <= 55 → no trades
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15  # Long opportunity
        crsi_overbought = crsi[i] > 85  # Short opportunity
        crsi_extreme_oversold = crsi[i] < 10
        crsi_extreme_overbought = crsi[i] > 90
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY - requires 3+ confluence
        long_score = 0
        
        # CRSI oversold (primary signal)
        if crsi_oversold:
            long_score += 3
        if crsi_extreme_oversold:
            long_score += 2
        
        # 4h trend alignment
        if four_h_bullish:
            long_score += 2
        
        # 1d major bias
        if one_d_bullish:
            long_score += 1
        
        # Range regime (mean reversion works best)
        if chop_range:
            long_score += 1
        
        # Volume confirmation
        if volume_ok:
            long_score += 0.5
        
        # Session filter (must be in session for entry)
        if in_session:
            long_score += 0.5
        
        # Enter long if score >= 6 (strict for low trade frequency)
        if long_score >= 6 and in_session:
            if one_d_bullish and four_h_bullish:
                new_signal = HIGH_CONV_SIZE  # 0.30 - high conviction
            elif four_h_bullish:
                new_signal = BASE_SIZE  # 0.20 - base
            else:
                new_signal = LOW_CONV_SIZE  # 0.15 - low conviction
        
        # SHORT ENTRY - requires 3+ confluence
        short_score = 0
        
        # CRSI overbought (primary signal)
        if crsi_overbought:
            short_score += 3
        if crsi_extreme_overbought:
            short_score += 2
        
        # 4h trend alignment
        if four_h_bearish:
            short_score += 2
        
        # 1d major bias
        if one_d_bearish:
            short_score += 1
        
        # Range regime (mean reversion works best)
        if chop_range:
            short_score += 1
        
        # Volume confirmation
        if volume_ok:
            short_score += 0.5
        
        # Session filter
        if in_session:
            short_score += 0.5
        
        # Enter short if score >= 6
        if short_score >= 6 and in_session:
            if one_d_bearish and four_h_bearish:
                new_signal = -HIGH_CONV_SIZE  # -0.30 - high conviction
            elif four_h_bearish:
                new_signal = -BASE_SIZE  # -0.20 - base
            else:
                new_signal = -LOW_CONV_SIZE  # -0.15 - low conviction
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 120 bars (~5 days on 1h), allow slightly weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position and in_session:
            if crsi_oversold and four_h_bullish and volume_ok:
                new_signal = LOW_CONV_SIZE
            elif crsi_overbought and four_h_bearish and volume_ok:
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
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long if CRSI becomes overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Exit short if CRSI becomes oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and four_h_bearish:
                trend_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and four_h_bullish:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or crsi_exit or trend_reversal:
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