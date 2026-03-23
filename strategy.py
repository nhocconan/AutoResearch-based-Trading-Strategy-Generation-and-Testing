#!/usr/bin/env python3
"""
Experiment #166: 12h Primary + 1d HTF — Regime-Adaptive CRSI/Choppiness Strategy

Hypothesis: Pure trend-following (Exp #164) failed because 2025 is bear/range market.
This strategy uses REGIME-ADAPTIVE logic proven in best performer (#163 Sharpe=0.486):

1) Choppiness Index (CHOP) regime detection:
   - CHOP > 61.8 = ranging market → use Connors RSI mean reversion
   - CHOP < 38.2 = trending market → use Donchian breakout + HMA trend
   
2) Connors RSI (CRSI) for mean reversion entries:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > SMA(200)
   - Short: CRSI > 90 + price < SMA(200)
   
3) 1d HMA(21) for macro bias filter (only trade WITH daily trend)

4) ATR(14) trailing stop at 2.5x for risk management

5) Position sizing: 0.25 base, 0.30 with full confluence

Why this should work:
- Regime detection prevents trend-following in chop (major whipsaw cause)
- CRSI has 75% win rate in backtests through 2022 crash
- 12h timeframe = 20-50 trades/year (optimal for fee drag)
- 1d HTF filter prevents counter-trend trades in strong macro moves

Target: Sharpe > 0.5 on ALL symbols, DD < -30%, 30+ trades on train
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_crsi_chop_donchian_1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging/choppy market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    # Streak = consecutive up (+1) or down (-1) days
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_series = pd.Series(streak)
    streak_gain = streak_series.clip(lower=0)
    streak_loss = (-streak_series).clip(lower=0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.values
    
    # Component 3: PercentRank of price change
    price_change = close_s.diff()
    percent_rank = pd.Series(price_change).rolling(window=pr_period, min_periods=pr_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= pr_period else np.nan
    ).values * 100.0
    
    # Combine all three components
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, period=200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_12h = calculate_hma(close, period=21)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_12h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION ===
        is_range_regime = chop_14[i] > 61.8
        is_trend_regime = chop_14[i] < 38.2
        # Neutral regime: 38.2 <= CHOP <= 61.8 (reduce position or stay flat)
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12h TREND ===
        price_above_hma_12h = close[i] > hma_12h[i]
        price_below_hma_12h = close[i] < hma_12h[i]
        
        # === RANGE REGIME: CONNORS RSI MEAN REVERSION ===
        crsi_oversold = crsi[i] < 15  # Slightly relaxed from 10 for more trades
        crsi_overbought = crsi[i] > 85  # Slightly relaxed from 90
        
        # === TREND REGIME: DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Range regime: CRSI mean reversion with macro filter
        if is_range_regime and crsi_oversold and price_above_hma_1d:
            new_signal = POSITION_SIZE_BASE
        
        # Range regime + 12h bullish: stronger signal
        if is_range_regime and crsi_oversold and price_above_hma_1d and price_above_hma_12h:
            new_signal = POSITION_SIZE_MAX
        
        # Trend regime: Donchian breakout with macro + 12h confirmation
        if is_trend_regime and breakout_long and price_above_hma_1d and price_above_hma_12h:
            new_signal = POSITION_SIZE_MAX
        
        # Trend regime: partial if only macro aligned
        if is_trend_regime and breakout_long and price_above_hma_1d:
            new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        # Range regime: CRSI mean reversion with macro filter
        if is_range_regime and crsi_overbought and price_below_hma_1d:
            new_signal = -POSITION_SIZE_BASE
        
        # Range regime + 12h bearish: stronger signal
        if is_range_regime and crsi_overbought and price_below_hma_1d and price_below_hma_12h:
            new_signal = -POSITION_SIZE_MAX
        
        # Trend regime: Donchian breakout with macro + 12h confirmation
        if is_trend_regime and breakout_short and price_below_hma_1d and price_below_hma_12h:
            new_signal = -POSITION_SIZE_MAX
        
        # Trend regime: partial if only macro aligned
        if is_trend_regime and breakout_short and price_below_hma_1d:
            new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought (range) or trend intact
                if (is_range_regime and crsi[i] < 80) or (is_trend_regime and price_above_hma_12h):
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold (range) or trend intact
                if (is_range_regime and crsi[i] > 20) or (is_trend_regime and price_below_hma_12h):
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        if in_position and position_side > 0:
            # Exit long if trend regime turns bearish
            if is_trend_regime and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend regime turns bullish
            if is_trend_regime and price_above_hma_12h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals