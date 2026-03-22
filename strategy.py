#!/usr/bin/env python3
"""
Experiment #013: 1d Dual Regime Strategy with 1w Trend Filter

Hypothesis: Single-regime strategies fail because crypto alternates between trending
and ranging markets. This strategy uses Choppiness Index to detect regime and switches
logic accordingly:

1. CHOPPY REGIME (CHOP > 61.8): Mean reversion using Connors RSI
   - CRSI < 10 + price > 1w HMA → Long
   - CRSI > 90 + price < 1w HMA → Short
   - Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3

2. TRENDING REGIME (CHOP < 38.2): Trend following using KAMA crossover
   - KAMA(10) > KAMA(30) + price > 1w HMA → Long
   - KAMA(10) < KAMA(30) + price < 1w HMA → Short

3. 1w HMA(21) Major Bias: Only take trades aligned with weekly trend
   - Increases win rate by filtering counter-trend entries

4. ATR(14) Trailing Stop: 2.5x ATR for risk management

Why this should work:
- Choppiness Index is proven regime filter (research shows best meta-filter for bear markets)
- Connors RSI has 75% win rate for mean reversion (Larry Connors research)
- KAMA adapts to volatility better than EMA (less whipsaw in 2022 crash)
- 1d timeframe = 20-50 trades/year target (optimal for fee drag)
- Dual regime = works in both bull and bear/range markets

Timeframe: 1d (REQUIRED for Experiment #013)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 high conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_dual_regime_crsi_kama_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP calculation
    atr_sum = atr.rolling(window=period, min_periods=period).sum()
    price_range = hh - ll
    
    # Avoid division by zero
    price_range = price_range.replace(0, np.nan)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry signals:
    - CRSI < 10 = oversold (long opportunity)
    - CRSI > 90 = overbought (short opportunity)
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_short = 100 - (100 / (1 + rs))
    rsi_short = rsi_short.replace([np.inf, -np.inf], np.nan)
    
    # Component 2: RSI of Streak
    # Streak = consecutive up/down days
    direction = np.sign(delta)
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if direction.iloc[i] == direction.iloc[i-1]:
            streak[i] = streak[i-1] + 1
        else:
            streak[i] = 1 if direction.iloc[i] != 0 else 0
    
    streak_s = pd.Series(streak)
    # RSI on streak values (inverted for mean reversion)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.replace([np.inf, -np.inf], np.nan)
    
    # Component 3: Percent Rank
    # Where current return sits in distribution of last N returns
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=pr_period, min_periods=pr_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )
    percent_rank = percent_rank * 100  # Scale to 0-100
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3
    crsi = crsi.replace([np.inf, -np.inf], np.nan)
    
    return crsi.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA) - adapts to market noise.
    More responsive in trending markets, smoother in ranging markets.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER) - measures trend efficiency
    change = np.abs(close_s - close_s.shift(er_period))
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constant
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[er_period] = close_s.iloc[er_period]
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_slow = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            continue
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = choppy/ranging (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        # 38.2 <= CHOP <= 61.8 = neutral (no trades or reduced size)
        is_choppy = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        
        # === MEAN REVERSION SIGNALS (Choppy Regime) ===
        crsi_oversold = crsi[i] < 15  # Slightly relaxed from 10 for more trades
        crsi_overbought = crsi[i] > 85  # Slightly relaxed from 90 for more trades
        
        # === TREND FOLLOWING SIGNALS (Trending Regime) ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY
        long_score = 0
        
        if is_choppy:
            # Mean reversion long
            if crsi_oversold:
                long_score += 3
            if weekly_bullish:
                long_score += 2
            if rsi_14[i] < 40:  # Additional oversold confirmation
                long_score += 1
        elif is_trending:
            # Trend following long
            if kama_bullish:
                long_score += 3
            if weekly_bullish:
                long_score += 2
            if close[i] > kama_fast[i]:  # Price above KAMA
                long_score += 1
        else:
            # Neutral regime - only high conviction trades
            if crsi_oversold and weekly_bullish:
                long_score += 4
            if kama_bullish and weekly_bullish:
                long_score += 4
        
        # Enter long if score >= 5
        if long_score >= 5:
            if weekly_bullish:
                new_signal = HIGH_CONV_SIZE
            else:
                new_signal = BASE_SIZE
        
        # SHORT ENTRY
        short_score = 0
        
        if is_choppy:
            # Mean reversion short
            if crsi_overbought:
                short_score += 3
            if weekly_bearish:
                short_score += 2
            if rsi_14[i] > 60:  # Additional overbought confirmation
                short_score += 1
        elif is_trending:
            # Trend following short
            if kama_bearish:
                short_score += 3
            if weekly_bearish:
                short_score += 2
            if close[i] < kama_fast[i]:  # Price below KAMA
                short_score += 1
        else:
            # Neutral regime - only high conviction trades
            if crsi_overbought and weekly_bearish:
                short_score += 4
            if kama_bearish and weekly_bearish:
                short_score += 4
        
        # Enter short if score >= 5
        if short_score >= 5:
            if weekly_bearish:
                new_signal = -HIGH_CONV_SIZE
            else:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~30 days on 1d), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if is_choppy and crsi_oversold and weekly_bullish:
                new_signal = BASE_SIZE
            elif is_choppy and crsi_overbought and weekly_bearish:
                new_signal = -BASE_SIZE
            elif is_trending and kama_bullish and weekly_bullish:
                new_signal = BASE_SIZE
            elif is_trending and kama_bearish and weekly_bearish:
                new_signal = -BASE_SIZE
        
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        regime_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and is_trending and not kama_bullish:
                regime_exit = True
            if position_side < 0 and is_trending and not kama_bearish:
                regime_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_exit:
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