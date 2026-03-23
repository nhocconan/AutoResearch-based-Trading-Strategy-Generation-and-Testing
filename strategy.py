#!/usr/bin/env python3
"""
Experiment #116: 12h Primary + 1d HTF — Dual Regime (Chop + Connors RSI + Donchian)

Hypothesis: Recent failures (#104-115) show complex regime detection adds lag, but pure
trend-following fails in bear/range markets (2025 BTC -25%). Research shows:
- Connors RSI has 75% win rate for mean-reversion in choppy markets
- Choppiness Index > 61.8 = range (use CRSI), < 38.2 = trend (use Donchian)
- 1d HMA filter prevents counter-trend trades in bear markets

This combines proven patterns:
1) Choppiness Index(14) regime detection — switch between mean-revert vs trend-follow
2) Connors RSI for range entries — (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3) Donchian(20) breakout for trend entries — price breaks 20-period high/low
4) 1d HMA(21) macro bias — only long if price > 1d HMA, only short if < 1d HMA
5) ATR(14) 2.5x trailing stop — limits drawdown on all positions

Why 12h should work:
- Higher TF = fewer trades (target 25-40/year) = less fee drag
- 1d HTF filter catches macro trend direction (critical for 2022 crash, 2025 bear)
- Dual regime adapts to market conditions (range vs trend)
- Simpler than failed #106 multi-signal confluence

Position size: 0.25 base, 0.30 max with strong confluence
Stoploss: 2.5*ATR trailing on all positions
Target: Sharpe > 0.5 on ALL symbols, 25-40 trades/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_donchian_1d_v1"
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
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days streak length
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (positive streak = high, negative = low)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period+1):i+1]
        if len(streak_window) > 0:
            avg_streak = np.mean(streak_window)
            # Normalize to 0-100 scale
            streak_rsi[i] = 50.0 + avg_streak * 10.0
            streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank component (today's return vs last 100 days)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[max(0, i-rank_period+1):i]
        if len(window) > 0:
            percentile = np.sum(window < returns[i]) / len(window)
            percent_rank[i] = percentile * 100.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        range_hl = hh - ll
        
        if range_hl > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    hma_12h_21 = calculate_hma(close, period=21)
    hma_12h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            continue
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 55.0  # Range/mean-reversion regime
        is_trending = choppiness[i] < 45.0  # Trend-following regime
        
        # === 12h TREND FILTER ===
        hma_12h_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_12h_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- MEAN REVERSION ENTRY (Choppy regime) ---
        # Long: CRSI < 15 (oversold) + price > 1d HMA (macro uptrend)
        # Short: CRSI > 85 (overbought) + price < 1d HMA (macro downtrend)
        if is_choppy:
            if crsi[i] < 15.0 and price_above_hma_1d:
                new_signal = POSITION_SIZE_BASE
            elif crsi[i] > 85.0 and price_below_hma_1d:
                new_signal = -POSITION_SIZE_BASE
        
        # --- TREND FOLLOWING ENTRY (Trending regime) ---
        # Long: Donchian breakout + price > 1d HMA + 12h HMA bullish
        # Short: Donchian breakdown + price < 1d HMA + 12h HMA bearish
        if is_trending:
            breakout_long = close[i] > donchian_upper[i-1]
            breakout_short = close[i] < donchian_lower[i-1]
            
            if breakout_long and price_above_hma_1d and hma_12h_bullish:
                new_signal = POSITION_SIZE_BASE
            elif breakout_short and price_below_hma_1d and hma_12h_bearish:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if regime hasn't switched against us
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if still in valid regime
                if (is_choppy and crsi[i] < 70.0) or (is_trending and close[i] > donchian_mid[i]):
                    if price_above_hma_1d:
                        new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if still in valid regime
                if (is_choppy and crsi[i] > 30.0) or (is_trending and close[i] < donchian_mid[i]):
                    if price_below_hma_1d:
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
        
        # === EXIT ON MACRO TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit in choppy regime) ===
        if in_position and position_side > 0 and is_choppy:
            if crsi[i] > 70.0:
                new_signal = 0.0
        
        if in_position and position_side < 0 and is_choppy:
            if crsi[i] < 30.0:
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