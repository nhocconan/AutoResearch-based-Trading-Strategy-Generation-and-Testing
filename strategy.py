#!/usr/bin/env python3
"""
Experiment #002: 12h Primary + 1d/1w HTF — Dual Regime with Connors RSI + Vol Spike

Hypothesis: Higher timeframes (12h) reduce noise and fee drag while maintaining
sufficient trade frequency. Combining Connors RSI (proven 75% win rate) with
Choppiness Index regime detection should work across bull/bear/range markets.

Key innovations:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. CHOPPINESS REGIME: CHOP(14) > 55 = range (mean revert), CHOP < 45 = trend
3. 1D HMA for intermediate trend, 1W HMA for secular trend bias
4. VOL SPIKE: ATR(7)/ATR(30) > 1.5 signals panic/reversion opportunity
5. LOOSE entry conditions to ensure ≥10 trades/symbol (critical requirement)

Why 12h works:
- Targets 20-50 trades/year (Rule 10 compliant)
- Less noise than 4h/1h, more signals than 1d
- Proven in research for BTC/ETH bear market performance

Entry conditions (LOOSE for trade generation):
- Long range: CRSI < 30 + CHOP > 50 + (price > 1d HMA OR vol spike)
- Short range: CRSI > 70 + CHOP > 50 + (price < 1d HMA OR vol spike)
- Long trend: CRSI < 50 + CHOP < 45 + 1d HMA bullish + 1w HMA bullish
- Short trend: CRSI > 50 + CHOP < 45 + 1d HMA bearish + 1w HMA bearish

Position size: 0.28 (discrete, Rule 4 compliant)
Stoploss: 2.5 * ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_dual_regime_1d1w_v1"
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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Streak calculation (consecutive up/down days)
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100) - percentile of today's return over last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period+1:i+1]
        if len(window_returns) > 0:
            current_return = returns.iloc[i]
            rank = np.sum(window_returns <= current_return) / len(window_returns)
            percent_rank[i] = rank * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for secular trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        if atr_14[i] == 0 or atr_30[i] == 0:
            continue
        
        # === VOLATILITY SPIKE FILTER ===
        vol_spike = (atr_7[i] / atr_30[i]) > 1.5  # High vol = reversion opportunity
        vol_normal = (atr_7[i] / atr_30[i]) <= 1.3  # Normal vol = trend following OK
        
        # === 1W SECULAR BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-5] if i >= 5 else True
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-5] if i >= 5 else False
        
        # === 1D INTERMEDIATE TREND ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else True
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        
        # === 12H SHORT-TERM TREND ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-2] if i >= 2 else True
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-2] if i >= 2 else False
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Lower threshold for more range trades
        is_trending = chop_value < 45.0  # Standard threshold
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 30  # Looser than 25 for more trades
        crsi_overbought = crsi[i] > 70  # Looser than 75 for more trades
        crsi_neutral_low = crsi[i] < 50
        crsi_neutral_high = crsi[i] > 50
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion (vol spike helps) ---
        if is_ranging:
            # Long: CRSI oversold + (near BB lower OR vol spike) + macro bias
            if crsi_oversold:
                if price_near_bb_lower or vol_spike:
                    if price_above_hma_1d or price_above_hma_1w or vol_spike:
                        new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + (near BB upper OR vol spike) + macro bias
            if new_signal == 0.0 and crsi_overbought:
                if price_near_bb_upper or vol_spike:
                    if price_below_hma_1d or price_below_hma_1w or vol_spike:
                        new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following ---
        elif is_trending:
            # Long: CRSI not overbought + 1d bullish + 1w bullish (any 2 of 3)
            if crsi_neutral_low:
                trend_score_long = (1 if hma_1d_slope_bull else 0) + (1 if price_above_hma_1d else 0) + (1 if hma_1w_slope_bull else 0)
                if trend_score_long >= 2:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI not oversold + 1d bearish + 1w bearish (any 2 of 3)
            if new_signal == 0.0 and crsi_neutral_high:
                trend_score_short = (1 if hma_1d_slope_bear else 0) + (1 if price_below_hma_1d else 0) + (1 if hma_1w_slope_bear else 0)
                if trend_score_short >= 2:
                    new_signal = -POSITION_SIZE
        
        # --- VOL SPIKE OVERRIDE: Always consider mean reversion on extreme vol ---
        if vol_spike and new_signal == 0.0:
            if crsi_oversold and price_near_bb_lower:
                new_signal = POSITION_SIZE
            elif crsi_overbought and price_near_bb_upper:
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime changes from ranging to strongly trending bearish
        if in_position and position_side > 0:
            if is_trending and hma_1d_slope_bear and price_below_hma_1w:
                new_signal = 0.0
        
        # Exit short if regime changes from ranging to strongly trending bullish
        if in_position and position_side < 0:
            if is_trending and hma_1d_slope_bull and price_above_hma_1w:
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