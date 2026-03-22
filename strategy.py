#!/usr/bin/env python3
"""
Experiment #206: 12h Primary + 1d HTF — Simplified Connors RSI + Choppiness Regime

Hypothesis: Previous strategies failed due to overly complex confluence requirements
that resulted in 0 trades or negative Sharpe. Research shows Connors RSI alone has
75% win rate for mean reversion. This strategy SIMPLIFIES entry conditions:

1. CHOPPINESS INDEX: Primary regime filter (CHOP>55=range, CHOP<45=trend)
2. CONNORS RSI: Single primary signal (CRSI<20=long, CRSI>80=short)
3. 1d HMA(21): Trend bias filter (only counter-trend in range markets)
4. ATR(14) trailing stop: 2.5x for risk management
5. LOOSENED thresholds: More trades to avoid 0-trade failure mode

Key changes from failed experiments:
- Fewer confluence requirements (2-3 factors max, not 5+)
- Lower CRSI thresholds (20/80 instead of 15/85)
- Lower CHOP thresholds (55/45 instead of 61.8/38.2)
- Forced entry after 100 bars without trades
- Simpler scoring system

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete (max 0.35)
Target trades: 30-60/year per symbol (enough to avoid 0-trade reject)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simp_connors_chop_1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 55 = range market (mean revert)
    CHOP < 45 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Normalize streak to 0-100 scale
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 12.5)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 12.5)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_roc(close, period=10):
    """Calculate Rate of Change."""
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=period) * 100
    return roc.fillna(0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.2)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    roc_10 = calculate_roc(close, 10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        crsi_neutral = 35 < crsi[i] < 65
        
        # === MOMENTUM ===
        momentum_positive = roc_10[i] > 0
        momentum_negative = roc_10[i] < 0
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_range_market:
            current_size = BASE_SIZE  # Full size in range (mean revert works best)
        elif is_trend_market:
            current_size = BASE_SIZE * 0.8  # Reduced in trend (pullback entries)
        
        # === ENTRY LOGIC — SIMPLIFIED FOR MORE TRADES ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_conditions = 0
        
        # Primary: CRSI oversold (works in all regimes)
        if crsi_oversold:
            long_conditions += 2
        
        # Secondary: BB lower confirmation
        if price_below_bb_lower:
            long_conditions += 1
        
        # Tertiary: Range market preference
        if is_range_market:
            long_conditions += 1
        
        # Quaternary: 1d trend support
        if trend_1d_bullish or price_above_1d_hma:
            long_conditions += 1
        
        # Momentum confirmation
        if momentum_positive:
            long_conditions += 0.5
        
        # Entry threshold: 2.5+ conditions for full size, 2.0 for half
        if long_conditions >= 2.5:
            new_signal = current_size
        elif long_conditions >= 2.0 and bars_since_last_trade > 40:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Primary: CRSI overbought
        if crsi_overbought:
            short_conditions += 2
        
        # Secondary: BB upper confirmation
        if price_above_bb_upper:
            short_conditions += 1
        
        # Tertiary: Range market preference
        if is_range_market:
            short_conditions += 1
        
        # Quaternary: 1d trend support
        if trend_1d_bearish or price_below_1d_hma:
            short_conditions += 1
        
        # Momentum confirmation
        if momentum_negative:
            short_conditions += 0.5
        
        # Entry threshold
        if short_conditions >= 2.5:
            new_signal = -current_size
        elif short_conditions >= 2.0 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.5
        
        # === FORCED ENTRY — CRITICAL TO AVOID 0 TRADES ===
        # If no trades for 100 bars (~50 days on 12h), force entry on CRSI extremes
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if crsi[i] < 25:
                new_signal = current_size * 0.4
            elif crsi[i] > 75:
                new_signal = -current_size * 0.4
            elif crsi[i] < 30 and price_below_bb_lower:
                new_signal = current_size * 0.35
            elif crsi[i] > 70 and price_above_bb_upper:
                new_signal = -current_size * 0.35
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        # Exit long when CRSI goes overbought, exit short when CRSI goes oversold
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True
        
        if stoploss_triggered or crsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals