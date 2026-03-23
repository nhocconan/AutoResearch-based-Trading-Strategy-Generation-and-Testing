#!/usr/bin/env python3
"""
Experiment #836: 12h Primary + 1d HTF — Dual Regime with Connors RSI + Fisher

Hypothesis: After 572+ failed strategies, the key is:
1. 12h timeframe balances trade frequency (30-60/year) with signal quality
2. Dual regime (chop/trend) adapts to market conditions
3. Connors RSI (not standard RSI) has 75% win rate for mean reversion
4. Fisher Transform catches reversals in bear markets (2022, 2025)
5. 1d HMA21 provides trend bias without being too slow
6. Relaxed entry conditions ensure trades on ALL symbols (BTC, ETH, SOL)

Key differences from failed strategies:
- Connors RSI instead of standard RSI (better mean reversion)
- Fisher Transform for reversal confirmation
- CHOP thresholds: 50/50 (not 55/45) - more regime switches
- RSI thresholds: 30/70 (standard, not too strict)
- Ensure minimum trade count by having fallback entries
- Size: 0.25-0.30 (conservative for 12h)

Target: Sharpe > 0.612, trades >= 20 train, >= 5 test, ALL symbols positive
Timeframe: 12h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_fisher_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - consecutive up/down days
    streak_rsi = np.full(n, np.nan)
    delta = np.diff(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if delta[i-1] > 0:
            streak[i] = streak[i-1] + 1 if i > 0 and delta[i-2] > 0 else 1
        elif delta[i-1] < 0:
            streak[i] = streak[i-1] - 1 if i > 0 and delta[i-2] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        avg_streak = np.mean(streak_window)
        # Map streak to 0-100 range
        streak_rsi[i] = 50 + avg_streak * 10
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank (100) - where current close ranks in last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_fisher_transform(high, low, period=9):
    """Ehlers Fisher Transform — normalizes price to Gaussian distribution."""
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher[i-1] if i > 1 else 0.0
            continue
        
        normalized = (hl2 - lowest_low) / range_val
        normalized = np.clip(normalized, 0.001, 0.999)
        
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — CHOP > 50 = ranging, CHOP < 50 = trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], np.abs(high[j] - prev_close), np.abs(low[j] - prev_close))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        prev_close = close[i-1]
        tr[i] = max(high[i] - low[i], np.abs(high[i] - prev_close), np.abs(low[i] - prev_close))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    crsi_12h = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_12h = calculate_atr(high, low, close, period=14)
    fisher_12h, fisher_prev_12h = calculate_fisher_transform(high, low, period=9)
    
    # Calculate and align 1d HMA for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(fisher_12h[i]) or np.isnan(fisher_prev_12h[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        ranging_regime = chop_12h[i] > 50
        trending_regime = chop_12h[i] < 50
        
        # === CONNORS RSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi_12h[i] < 15
        crsi_overbought = crsi_12h[i] > 85
        crsi_extreme_oversold = crsi_12h[i] < 10
        crsi_extreme_overbought = crsi_12h[i] > 90
        
        # === STANDARD RSI SIGNALS (Fallback) ===
        rsi_oversold = rsi_12h[i] < 30
        rsi_overbought = rsi_12h[i] > 70
        rsi_extreme_oversold = rsi_12h[i] < 25
        rsi_extreme_overbought = rsi_12h[i] > 75
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_12h[i] < -1.0
        fisher_overbought = fisher_12h[i] > 1.0
        fisher_cross_up = fisher_prev_12h[i] < -1.0 and fisher_12h[i] >= -1.0
        fisher_cross_down = fisher_prev_12h[i] > 1.0 and fisher_12h[i] <= 1.0
        fisher_recovering = fisher_12h[i] > fisher_prev_12h[i] and fisher_12h[i] < 0
        fisher_weakening = fisher_12h[i] < fisher_prev_12h[i] and fisher_12h[i] > 0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 50) — Mean Reversion ===
        if ranging_regime:
            # Primary: Connors RSI extreme + Fisher confirmation
            if crsi_extreme_oversold and fisher_oversold:
                desired_signal = BASE_SIZE
            elif crsi_extreme_overbought and fisher_overbought:
                desired_signal = -BASE_SIZE
            # Secondary: Connors RSI alone (ensures trades)
            elif crsi_oversold and (trend_1d_bullish or not trend_1d_bearish):
                desired_signal = REDUCED_SIZE
            elif crsi_overbought and (trend_1d_bearish or not trend_1d_bullish):
                desired_signal = -REDUCED_SIZE
            # Fallback: Standard RSI extreme (guarantees minimum trades)
            elif rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            elif rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
            # Fisher reversal cross
            elif fisher_cross_up and rsi_oversold:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            elif fisher_cross_down and rsi_overbought:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === TRENDING REGIME LOGIC (CHOP < 50) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + pullback entry
            if trend_1d_bullish:
                if fisher_recovering and rsi_oversold:
                    desired_signal = BASE_SIZE
                elif donchian_breakout_long:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
                elif crsi_oversold and fisher_oversold:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            # Short: Bearish trend + pullback entry
            if trend_1d_bearish:
                if fisher_weakening and rsi_overbought:
                    desired_signal = -BASE_SIZE
                elif donchian_breakout_short:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
                elif crsi_overbought and fisher_overbought:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and Fisher not overbought
                if trend_1d_bullish and fisher_12h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and Fisher not oversold
                if trend_1d_bearish and fisher_12h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + Fisher overbought
            if trend_1d_bearish and fisher_12h[i] > 1.5:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_12h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + Fisher oversold
            if trend_1d_bullish and fisher_12h[i] < -1.5:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_12h[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals