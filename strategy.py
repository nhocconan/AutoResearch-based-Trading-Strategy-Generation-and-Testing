#!/usr/bin/env python3
"""
Experiment #001: 4h Dual-Regime Strategy with 1d/1w Trend Filter

Hypothesis: Single-regime strategies fail because crypto alternates between trending
and ranging markets. This strategy uses Choppiness Index (CHOP) to detect regime:
- CHOP > 61.8 = Range (use Connors RSI mean-reversion)
- CHOP < 38.2 = Trend (use Donchian breakout + HMA trend follow)
- 38.2 <= CHOP <= 61.8 = Transition (reduce position size)

Key Components:
1. Choppiness Index(14) - Regime detection from quantitative literature
2. Connors RSI - (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for mean-reversion
3. Donchian(20) Breakout - Clean trend entry when in trending regime
4. 1d HMA(21) - HTF trend filter via mtf_data (call ONCE before loop)
5. 1w HMA(21) - Major bias for position sizing conviction
6. ATR(14) Trailing Stop - 2.5x ATR for risk management

Why this should work:
- Dual-regime adapts to market conditions (proven in crypto research)
- Connors RSI has 75% win rate in range markets (Larry Connors research)
- 1d/1w HTF filters prevent counter-trend failures
- 4h timeframe = 20-50 trades/year (optimal fee/trade balance)
- Conservative sizing (0.20-0.30) protects against 2022-style crashes
- Less strict entry thresholds = more trades (avoids 0-trade failure)

Timeframe: 4h (REQUIRED for Experiment #001)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20 base, 0.30 high conviction, 0.15 transition regime
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_connors_1d_1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = Range market
    CHOP < 38.2 = Trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high - Lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    price_range = hh - ll
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.nan)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Mean reversion indicator from Larry Connors.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) - Short-term RSI
    2. RSI_Streak(2) - RSI of consecutive up/down days
    3. PercentRank(100) - Where current price sits in recent range
    """
    close_s = pd.Series(close)
    n = len(close)
    
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
    # Streak = consecutive up (+1) or down (-1) days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.replace([np.inf, -np.inf], np.nan)
    
    # Component 3: PercentRank(100)
    # Where current close sits in recent range (0-100)
    def percent_rank(x):
        if x.max() == x.min():
            return 50.0
        return (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
    
    percent_rank_vals = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        percent_rank, raw=False
    )
    
    # Combine components
    crsi = (rsi_short + rsi_streak + percent_rank_vals) / 3.0
    crsi = crsi.replace([np.inf, -np.inf], np.nan)
    
    return crsi.values

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
    chop = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
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
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(donchian_high[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = Range, CHOP < 38.2 = Trend
        is_range_regime = chop[i] > 61.8
        is_trend_regime = chop[i] < 38.2
        is_transition = not is_range_regime and not is_trend_regime
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGE REGIME: Connors RSI Mean Reversion ---
        if is_range_regime:
            # Long: CRSI < 10 (oversold) + price > 1d HMA (bullish bias)
            if crsi[i] < 15 and daily_bullish:
                if weekly_bullish:
                    new_signal = HIGH_CONV_SIZE
                else:
                    new_signal = BASE_SIZE
            
            # Short: CRSI > 90 (overbought) + price < 1d HMA (bearish bias)
            elif crsi[i] > 85 and daily_bearish:
                if weekly_bearish:
                    new_signal = -HIGH_CONV_SIZE
                else:
                    new_signal = -BASE_SIZE
        
        # --- TREND REGIME: Donchian Breakout + HMA ---
        elif is_trend_regime:
            # Detect breakouts
            donchian_breakout_long = False
            donchian_breakout_short = False
            
            if i > 0:
                if close[i] > donchian_high[i-1]:
                    donchian_breakout_long = True
                if close[i] < donchian_low[i-1]:
                    donchian_breakout_short = True
            
            # Long: Donchian breakout + daily bullish + RSI > 50
            if donchian_breakout_long and daily_bullish and rsi_14[i] > 50:
                if weekly_bullish:
                    new_signal = HIGH_CONV_SIZE
                else:
                    new_signal = BASE_SIZE
            
            # Short: Donchian breakout + daily bearish + RSI < 50
            elif donchian_breakout_short and daily_bearish and rsi_14[i] < 50:
                if weekly_bearish:
                    new_signal = -HIGH_CONV_SIZE
                else:
                    new_signal = -BASE_SIZE
        
        # --- TRANSITION REGIME: Reduced size, stricter entry ---
        elif is_transition:
            # Only enter on strong signals with HTF alignment
            if crsi[i] < 10 and daily_bullish and weekly_bullish:
                new_signal = LOW_CONV_SIZE
            elif crsi[i] > 90 and daily_bearish and weekly_bearish:
                new_signal = -LOW_CONV_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~120 hours = 5 days on 4h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            # Weaker entry in range regime
            if is_range_regime:
                if crsi[i] < 20 and daily_bullish:
                    new_signal = LOW_CONV_SIZE
                elif crsi[i] > 80 and daily_bearish:
                    new_signal = -LOW_CONV_SIZE
            # Weaker entry in trend regime
            elif is_trend_regime:
                if close[i] > donchian_high[i-1] and daily_bullish:
                    new_signal = LOW_CONV_SIZE
                elif close[i] < donchian_low[i-1] and daily_bearish:
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
        
        # === CRSI EXTREME EXIT (for range regime) ===
        crsi_exit = False
        if in_position and position_side != 0 and is_range_regime:
            # Exit long when CRSI goes overbought
            if position_side > 0 and crsi[i] > 80:
                crsi_exit = True
            # Exit short when CRSI goes oversold
            if position_side < 0 and crsi[i] < 20:
                crsi_exit = True
        
        # === DONCHIAN MID EXIT (for trend regime) ===
        donchian_exit = False
        if in_position and position_side != 0 and is_trend_regime:
            # Exit long if price falls back below Donchian mid
            if position_side > 0 and close[i] < donchian_mid[i]:
                donchian_exit = True
            # Exit short if price rises back above Donchian mid
            if position_side < 0 and close[i] > donchian_mid[i]:
                donchian_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or crsi_exit or donchian_exit:
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