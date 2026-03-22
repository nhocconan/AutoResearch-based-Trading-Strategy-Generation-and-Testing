#!/usr/bin/env python3
"""
Experiment #064: 4h Primary + 12h/1d HTF — Dual Regime Strategy

Hypothesis: Previous strategies failed because they used ONE approach (trend OR mean-revert)
throughout all market conditions. This strategy SWITCHES logic based on Choppiness Index:

REGIME DETECTION (Choppiness Index 14):
- CHOP > 61.8 = RANGING market → use Connors RSI mean reversion
- CHOP < 38.2 = TRENDING market → use HMA crossover trend following
- 38.2 <= CHOP <= 61.8 = TRANSITION → reduce position size by 50%

ENTRY LOGIC BY REGIME:
1. TRENDING (CHOP < 38.2):
   - Long: 12h HMA slope > 0 + 4h HMA(8) > HMA(21) + RSI(14) > 45
   - Short: 12h HMA slope < 0 + 4h HMA(8) < HMA(21) + RSI(14) < 55
   
2. RANGING (CHOP > 61.8):
   - Long: Connors RSI < 15 + price > 1d HMA(21) (mean reversion with trend bias)
   - Short: Connors RSI > 85 + price < 1d HMA(21)

3. TRANSITION (38.2 <= CHOP <= 61.8):
   - Only take strongest signals, reduce size to 0.15

HTF BIAS (12h + 1d):
- 12h HMA(21) slope determines primary bias
- 1d HMA(21) slope confirms major trend
- Only enter trades aligned with HTF bias (reduces counter-trend losses)

RISK MANAGEMENT:
- ATR(14) trailing stop at 2.5x (wider for 4h timeframe)
- Position size: 0.30 normal, 0.15 transition, 0.0 exit
- Max 40-60 trades/year on 4h timeframe

Why this should beat previous attempts:
- Regime switching adapts to market conditions (trend vs range)
- 4h provides better trade frequency than 12h/1d while avoiding fee drag
- HTF bias prevents counter-trend trades in strong moves
- Connors RSI proven effective for mean reversion in ranges
- Discrete sizing minimizes fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (0.0, ±0.15, ±0.30)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_connors_hma_12h1d_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = Market is choppy/ranging (mean reversion preferred)
    - CHOP < 38.2 = Market is trending (trend following preferred)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in the lookback
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        # Highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) of close - short-term momentum
    2. RSI(2) of streak - streak duration (consecutive up/down days)
    3. PercentRank(100) - where current price ranks vs last 100 bars
    
    Interpretation:
    - CRSI < 10 = Oversold (potential long)
    - CRSI > 90 = Overbought (potential short)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # Component 1: RSI(3) of close
    rsi_3 = calculate_rsi(close, 3)
    
    # Component 2: RSI(2) of streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (positive streak = gain, negative = loss)
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine components
    for i in range(max(3, streak_period, rank_period), n):
        crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # HMA for trend following
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    TRANSITION_SIZE = 0.15
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        
        if chop_value > 61.8:
            regime = 'range'  # Mean reversion
        elif chop_value < 38.2:
            regime = 'trend'  # Trend following
        else:
            regime = 'transition'  # Uncertain
        
        # === HTF TREND BIAS (12h + 1d) ===
        # 12h slope determines primary bias
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0
        trend_12h_bearish = hma_12h_slope_aligned[i] < 0
        
        # 1d slope confirms major trend
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0
        trend_1d_bearish = hma_1d_slope_aligned[i] < 0
        
        # Price vs 1d HMA for additional confirmation
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # Strong bias when both 12h and 1d agree
        strong_bullish = trend_12h_bullish and trend_1d_bullish
        strong_bearish = trend_12h_bearish and trend_1d_bearish
        
        # === POSITION SIZING BY REGIME ===
        if regime == 'trend':
            current_size = BASE_SIZE
        elif regime == 'range':
            current_size = BASE_SIZE
        else:  # transition
            current_size = TRANSITION_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # --- TRENDING REGIME (CHOP < 38.2) ---
        if regime == 'trend':
            # LONG: 12h bullish + HMA(8) > HMA(21) + RSI > 45
            if strong_bullish or (trend_12h_bullish and price_above_1d_hma):
                if hma_8[i] > hma_21[i] and rsi_14[i] > 45:
                    # Check for HMA crossover for stronger entry
                    hma_cross_bull = hma_8[i] > hma_21[i] and hma_8[i-1] <= hma_21[i-1]
                    if hma_cross_bull:
                        new_signal = current_size
                    elif rsi_14[i] > 50 and rsi_14[i] < 70:
                        new_signal = current_size * 0.8
            
            # SHORT: 12h bearish + HMA(8) < HMA(21) + RSI < 55
            if strong_bearish or (trend_12h_bearish and price_below_1d_hma):
                if hma_8[i] < hma_21[i] and rsi_14[i] < 55:
                    # Check for HMA crossover for stronger entry
                    hma_cross_bear = hma_8[i] < hma_21[i] and hma_8[i-1] >= hma_21[i-1]
                    if hma_cross_bear:
                        new_signal = -current_size
                    elif rsi_14[i] > 30 and rsi_14[i] < 50:
                        new_signal = -current_size * 0.8
        
        # --- RANGING REGIME (CHOP > 61.8) ---
        elif regime == 'range':
            # LONG: Connors RSI < 15 + price > 1d HMA (mean reversion with trend bias)
            if crsi[i] < 15 and price_above_1d_hma:
                new_signal = current_size
            elif crsi[i] < 20 and trend_12h_bullish:
                new_signal = current_size * 0.8
            
            # SHORT: Connors RSI > 85 + price < 1d HMA
            if crsi[i] > 85 and price_below_1d_hma:
                new_signal = -current_size
            elif crsi[i] > 80 and trend_12h_bearish:
                new_signal = -current_size * 0.8
        
        # --- TRANSITION REGIME (38.2 <= CHOP <= 61.8) ---
        else:  # transition
            # Only take strongest signals with reduced size
            if strong_bullish and hma_8[i] > hma_21[i] and rsi_14[i] > 55:
                new_signal = TRANSITION_SIZE
            elif strong_bearish and hma_8[i] < hma_21[i] and rsi_14[i] < 45:
                new_signal = -TRANSITION_SIZE
            elif crsi[i] < 10:
                new_signal = TRANSITION_SIZE * 0.8
            elif crsi[i] > 90:
                new_signal = -TRANSITION_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~33 days on 4h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if strong_bullish and hma_8[i] > hma_21[i]:
                new_signal = TRANSITION_SIZE
            elif strong_bearish and hma_8[i] < hma_21[i]:
                new_signal = -TRANSITION_SIZE
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h trend reverses bearish
            if position_side > 0 and trend_12h_bearish and hma_8[i] < hma_21[i]:
                trend_reversal = True
            # Exit short if 12h trend reverses bullish
            if position_side < 0 and trend_12h_bullish and hma_8[i] > hma_21[i]:
                trend_reversal = True
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes dramatically against position
        regime_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and chop_value > 70:  # Strong chop against long
                regime_exit = True
            if position_side < 0 and chop_value < 30:  # Strong trend against short
                regime_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or regime_exit:
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