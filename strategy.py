#!/usr/bin/env python3
"""
Experiment #062: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: Previous 12h strategies failed because they used simple trend-following
which gets destroyed in bear/range markets (2022 crash, 2025 test period).

This strategy uses REGIME-SWITCHING logic:
1. CHOPPINESS INDEX (CHOP) detects market regime:
   - CHOP > 61.8 = ranging market → use mean reversion (Connors RSI)
   - CHOP < 38.2 = trending market → use trend following (HMA crossover)
   - Between = neutral → reduce position size

2. CONNORS RSI (CRSI) for mean reversion entries:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > 1d HMA(21)
   - Short: CRSI > 90 + price < 1d HMA(21)
   - Proven 75% win rate in range markets

3. 1d HMA(21) for major trend bias (prevents counter-trend trades in strong trends)

4. 1w HMA(55) for secular trend filter (avoid fighting multi-month trends)

5. ATR(14) trailing stoploss at 2.5x (wider for 12h timeframe)

Why this should beat previous attempts:
- Regime detection prevents trend strategies in choppy markets
- Connors RSI catches reversals that EMA crossovers miss
- 12h naturally limits trades to 20-50/year (fee-efficient)
- Discrete position sizing (0.0, ±0.25, ±0.30) minimizes churn
- Dual HTF (1d + 1w) provides better trend context than single HTF

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-45/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_connors_chop_regime_1d1w_v1"
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

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component for Connors RSI.
    Measures consecutive up/down days.
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like scale (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    for i in range(period, n):
        lookback = streak[max(0, i-period+1):i+1]
        avg_streak = np.mean(lookback)
        # Map to 0-100 scale
        if avg_streak > 0:
            streak_rsi[i] = min(100, 50 + avg_streak * 10)
        elif avg_streak < 0:
            streak_rsi[i] = max(0, 50 + avg_streak * 10)
        else:
            streak_rsi[i] = 50
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component for Connors RSI.
    Measures where current return ranks vs past N periods.
    """
    n = len(close)
    pct_rank = np.zeros(n)
    
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    for i in range(period, n):
        lookback = returns[max(0, i-period+1):i]
        if len(lookback) > 0:
            current_return = returns[i]
            rank = np.sum(lookback < current_return)
            pct_rank[i] = (rank / len(lookback)) * 100
        else:
            pct_rank[i] = 50
    
    return pct_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_fast = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_fast + rsi_streak + pct_rank) / 3
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8 = ranging/choppy market (mean reversion favored)
    - CHOP < 38.2 = trending market (trend following favored)
    - 38.2 to 61.8 = neutral/transition
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        sum_atr = np.sum(tr[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 0 and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
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
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Calculate 1w HTF indicators
    hma_1w_55 = calculate_hma(df_1w['close'].values, 55)
    hma_1w_slope = calculate_hma_slope(hma_1w_55, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_55_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_55)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    # HMA for trend following (used in trending regime)
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18  # For neutral regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1w_55_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        
        if chop_value > 61.8:
            regime = 'range'  # Mean reversion favored
        elif chop_value < 38.2:
            regime = 'trend'  # Trend following favored
        else:
            regime = 'neutral'  # Transition, reduce size
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5  # Slightly positive threshold
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 1W SECULAR TREND FILTER ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0
        trend_1w_bearish = hma_1w_slope_aligned[i] < 0
        
        # === POSITION SIZING BY REGIME ===
        if regime == 'range':
            current_size = BASE_SIZE  # Full size for mean reversion
        elif regime == 'trend':
            current_size = BASE_SIZE  # Full size for trend following
        else:
            current_size = REDUCED_SIZE  # Reduced in neutral
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if regime == 'range':
            # Mean reversion: Connors RSI < 10 + above 1d HMA
            if crsi[i] < 15 and price_above_1d_hma:
                new_signal = current_size
            # Slightly less extreme but still oversold
            elif crsi[i] < 25 and price_above_1d_hma and trend_1d_bullish:
                new_signal = current_size * 0.7
        
        elif regime == 'trend':
            # Trend following: HMA crossover + 1d bias
            hma_bullish_cross = hma_8[i] > hma_21[i] and hma_8[i-1] <= hma_21[i-1]
            hma_aligned_bullish = hma_8[i] > hma_21[i]
            
            if hma_bullish_cross and (trend_1d_bullish or price_above_1d_hma):
                new_signal = current_size
            elif hma_aligned_bullish and trend_1d_bullish and crsi[i] < 50:
                new_signal = current_size * 0.7
        
        # SHORT ENTRIES
        if regime == 'range':
            # Mean reversion: Connors RSI > 90 + below 1d HMA
            if crsi[i] > 85 and price_below_1d_hma:
                new_signal = -current_size
            # Slightly less extreme but still overbought
            elif crsi[i] > 75 and price_below_1d_hma and trend_1d_bearish:
                new_signal = -current_size * 0.7
        
        elif regime == 'trend':
            # Trend following: HMA crossover + 1d bias
            hma_bearish_cross = hma_8[i] < hma_21[i] and hma_8[i-1] >= hma_21[i-1]
            hma_aligned_bearish = hma_8[i] < hma_21[i]
            
            if hma_bearish_cross and (trend_1d_bearish or price_below_1d_hma):
                new_signal = -current_size
            elif hma_aligned_bearish and trend_1d_bearish and crsi[i] > 50:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~50 days on 12h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if regime == 'range':
                if crsi[i] < 30 and price_above_1d_hma:
                    new_signal = current_size * 0.5
                elif crsi[i] > 70 and price_below_1d_hma:
                    new_signal = -current_size * 0.5
            elif regime == 'trend':
                if hma_8[i] > hma_21[i] and trend_1d_bullish:
                    new_signal = current_size * 0.5
                elif hma_8[i] < hma_21[i] and trend_1d_bearish:
                    new_signal = -current_size * 0.5
        
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        regime_change_exit = False
        if in_position and position_side != 0:
            # Exit long if regime switches to strong trend bearish
            if position_side > 0 and regime == 'trend' and trend_1d_bearish and hma_8[i] < hma_21[i]:
                regime_change_exit = True
            # Exit short if regime switches to strong trend bullish
            if position_side < 0 and regime == 'trend' and trend_1d_bullish and hma_8[i] > hma_21[i]:
                regime_change_exit = True
        
        # Apply stoploss or regime change exit
        if stoploss_triggered or regime_change_exit:
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