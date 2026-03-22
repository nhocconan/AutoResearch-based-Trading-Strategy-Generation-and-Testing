#!/usr/bin/env python3
"""
Experiment #191: 4h Primary + 1d/1w HTF — Regime-Adaptive Mean Reversion + Trend

Hypothesis: Recent failures (exp 179-190) show 0 trades due to overly strict filters.
This strategy SIMPLIFIES entry logic while maintaining edge:

1. CONNORS RSI (CRSI): Primary entry trigger - loosened thresholds (35/65 vs 25/75)
2. CHOPPINESS INDEX: Regime filter but permissive (50 threshold, not 55/45)
3. 1d HMA(21): Trend bias only - not hard filter, just size adjustment
4. 1w HMA(84): Major trend context for position sizing
5. ATR(14) trailing stop: 2.5x for risk management

Key changes from failed strategies:
- CRSI thresholds: 35/65 (was 25/75) - MORE TRADES
- CHOP threshold: 50 (was 55/45 dual) - SIMPLER
- No vol_spike requirement (was killing trades)
- Score-based entry: >=1 triggers (was >=2)
- Minimum 30 bars between trades (was 50-80)

Why this works for 4h:
- 4h has natural mean-reversion tendencies
- 1d HTF prevents fighting major trends
- Looser filters = 30-60 trades/year target (within 4h limits)
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Timeframe: 4h (REQUIRED for exp 191)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 max, discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_connors_chop_regime_1d1w_v2"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    Using 50 as neutral threshold for simpler logic.
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_values = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
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
    Lowered thresholds for more trades: 35/65 instead of 25/75
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
    
    # Convert streak to 0-100 scale
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
    """Calculate HMA slope as percentage change over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Calculate 1w HTF indicators
    hma_1w_84 = calculate_hma(df_1w['close'].values, 84)
    hma_1w_slope = calculate_hma_slope(hma_1w_84, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_84_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_84)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Bollinger Bands for additional confirmation
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
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
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1D TREND BIAS (soft filter - adjusts size, not hard block) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.15
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.15
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 1W MAJOR TREND (for sizing) ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.1 if not np.isnan(hma_1w_slope_aligned[i]) else False
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.1 if not np.isnan(hma_1w_slope_aligned[i]) else False
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 50
        is_trend_market = chop_14[i] < 50
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        near_bb_lower = close[i] < bb_lower[i] * 1.005
        near_bb_upper = close[i] > bb_upper[i] * 0.995
        
        # === CONNORS RSI (LOOSENED THRESHOLDS FOR MORE TRADES) ===
        crsi_oversold = crsi[i] < 35
        crsi_overbought = crsi[i] > 65
        crsi_extreme_low = crsi[i] < 25
        crsi_extreme_high = crsi[i] > 75
        crsi_neutral_low = crsi[i] < 40
        crsi_neutral_high = crsi[i] > 60
        
        # === POSITION SIZING ADJUSTMENT ===
        current_size = BASE_SIZE
        
        # Reduce size in neutral 1d trend
        if trend_1d_neutral:
            current_size = BASE_SIZE * 0.8
        
        # Increase size when 1w aligns with entry
        if trend_1w_bullish and trend_1d_bullish:
            current_size = min(0.35, BASE_SIZE * 1.1)
        if trend_1w_bearish and trend_1d_bearish:
            current_size = min(0.35, BASE_SIZE * 1.1)
        
        # === ENTRY LOGIC (SIMPLIFIED - score >= 1 triggers) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_score = 0
        
        # Path 1: CRSI oversold + near BB lower (primary mean revert)
        if crsi_oversold and (price_below_bb_lower or near_bb_lower):
            long_score += 2
        
        # Path 2: Range market + CRSI oversold
        if is_range_market and crsi_oversold:
            long_score += 2
        
        # Path 3: 1d bullish + CRSI pullback
        if trend_1d_bullish and crsi_neutral_low and price_above_1d_hma:
            long_score += 1
        
        # Path 4: Extreme oversold (always take)
        if crsi_extreme_low:
            long_score += 2
        
        # Path 5: 1w bullish major trend + any oversold
        if trend_1w_bullish and crsi[i] < 45:
            long_score += 1
        
        # Trigger long if score >= 1 and cooldown passed
        if long_score >= 1 and bars_since_last_trade > 30:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 20:
            new_signal = current_size
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: CRSI overbought + near BB upper
        if crsi_overbought and (price_above_bb_upper or near_bb_upper):
            short_score += 2
        
        # Path 2: Range market + CRSI overbought
        if is_range_market and crsi_overbought:
            short_score += 2
        
        # Path 3: 1d bearish + CRSI rally
        if trend_1d_bearish and crsi_neutral_high and price_below_1d_hma:
            short_score += 1
        
        # Path 4: Extreme overbought (always take)
        if crsi_extreme_high:
            short_score += 2
        
        # Path 5: 1w bearish major trend + any overbought
        if trend_1w_bearish and crsi[i] > 55:
            short_score += 1
        
        # Trigger short if score >= 1 and cooldown passed
        if short_score >= 1 and bars_since_last_trade > 30:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 20:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            # Force entry based on simple CRSI extremes
            if crsi[i] < 30:
                new_signal = HALF_SIZE
            elif crsi[i] > 70:
                new_signal = -HALF_SIZE
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d turns strongly bearish
            if position_side > 0 and hma_1d_slope_aligned[i] < -0.3:
                regime_reversal = True
            # Exit short if 1d turns strongly bullish
            if position_side < 0 and hma_1d_slope_aligned[i] > 0.3:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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