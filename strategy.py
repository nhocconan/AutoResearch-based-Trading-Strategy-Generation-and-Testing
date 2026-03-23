#!/usr/bin/env python3
"""
Experiment #001: 4h Primary + 1d/1w HTF — Choppiness Index + Connors RSI + HMA Trend

Hypothesis: After 607+ failed strategies, the clearest pattern is:
1. 4h timeframe produces 20-50 trades/year (optimal fee/trade balance)
2. Choppiness Index (CHOP) regime detection works exceptionally well on ETH (Sharpe +0.923)
3. Connors RSI (CRSI) mean reversion has 75% win rate in literature
4. HMA provides faster trend signal than EMA with less lag
5. Dual regime: mean-revert when CHOP>61.8 (range), trend-follow when CHOP<38.2 (trend)

This strategy uses:
- Choppiness Index (14) for regime: >61.8=range, <38.2=trend
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- HMA(21) on 4h for trend bias
- 1d HMA(50) for higher-timeframe trend confirmation
- ATR(14) for stoploss (2.5*ATR trailing)

Why this might beat Sharpe=0.520:
- CHOP regime filter prevents trend-following in chop (major 2022-2024 loss source)
- CRSI more sensitive than standard RSI for mean reversion
- 4h primary = fewer trades than 1h, less fee drag
- 1d HTF confirms major trend direction (avoid counter-trend trades)
- Asymmetric: mean-revert in range, trend-pullback in trend regime

Position sizing: 0.30 discrete (conservative for 4h TF per Rule 4)
Target: 25-45 trades/year on 4h
Stoploss: 2.5*ATR trailing via signal→0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_hma_1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile of today's change vs last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period_rsi)
    
    # Component 2: Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(period_streak, n):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak_abs[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 - streak_abs[i] * 10)
    
    # Component 3: Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(period_rank, n):
        returns = np.diff(close[i-period_rank:i+1])
        if len(returns) > 0 and returns[-1] != 0:
            percentile = np.sum(returns[:-1] <= returns[-1]) / (len(returns) - 1)
            percent_rank[i] = percentile * 100
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Range/consolidation (mean reversion favorable)
    - CHOP < 38.2: Trend (trend following favorable)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness calculation
    range_hl = hh - ll
    chop = 100.0 * np.log10(atr_sum / (range_hl + 1e-10)) / np.log10(period)
    
    return chop.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for higher-timeframe trend
    hma_1d = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    HALF_POSITION = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    profit_target_hit = False
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range_regime = chop[i] > 61.8
        is_trend_regime = chop[i] < 38.2
        
        # === 4H TREND BIAS ===
        hma_21_slope_bull = hma_21[i] > hma_21[i-5] if i >= 5 else False
        hma_21_slope_bear = hma_21[i] < hma_21[i-5] if i >= 5 else False
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        
        # === 1D TREND CONFIRMATION ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        price_below_hma_1d = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15  # Strong mean-reversion long signal
        crsi_overbought = crsi[i] > 85  # Strong mean-reversion short signal
        crsi_moderate_low = crsi[i] < 30  # Pullback long in uptrend
        crsi_moderate_high = crsi[i] > 70  # Pullback short in downtrend
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (CHOP > 61.8) + CRSI oversold
        if is_range_regime:
            if crsi_oversold:
                new_signal = POSITION_SIZE
        
        # Regime 2: Trending market (CHOP < 38.2) + 4h bull + CRSI pullback
        elif is_trend_regime:
            if hma_21_slope_bull and price_above_hma_21:
                if crsi_moderate_low and price_above_hma_1d:
                    new_signal = POSITION_SIZE
        
        # Regime 3: Neutral (38.2 <= CHOP <= 61.8) + strong HMA trend + CRSI extreme
        else:
            if hma_21_slope_bull and price_above_hma_21 and crsi_oversold:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (CHOP > 61.8) + CRSI overbought
        if is_range_regime:
            if crsi_overbought:
                new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market (CHOP < 38.2) + 4h bear + CRSI pullback
        elif is_trend_regime:
            if hma_21_slope_bear and price_below_hma_21:
                if crsi_moderate_high and price_below_hma_1d:
                    new_signal = -POSITION_SIZE
        
        # Regime 3: Neutral + strong HMA trend + CRSI extreme
        else:
            if hma_21_slope_bear and price_below_hma_21 and crsi_overbought:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === TAKE PROFIT (reduce to half at 2R) ===
        if in_position and not profit_target_hit:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * 2.5 * atr_14[i]:  # 2R profit
                    new_signal = HALF_POSITION
                    profit_target_hit = True
            elif position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * 2.5 * atr_14[i]:  # 2R profit
                    new_signal = -HALF_POSITION
                    profit_target_hit = True
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_21_slope_bear and price_below_hma_21:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_21_slope_bull and price_above_hma_21:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                profit_target_hit = False
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                profit_target_hit = False
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                profit_target_hit = False
        
        signals[i] = new_signal
    
    return signals