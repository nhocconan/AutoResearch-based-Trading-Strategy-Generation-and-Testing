#!/usr/bin/env python3
"""
Experiment #354: 4h Primary + 12h/1d HTF — Connors RSI + Donchian Breakout with Dual HMA Bias

Hypothesis: Previous Fisher-based strategies (#351) had modest Sharpe (0.228) because:
1. Fisher Transform extremes are rare on 4h, limiting trade frequency
2. Choppiness Index regime switching added complexity without clear benefit
3. Entry conditions too strict for consistent triggers

This strategy uses CONNORS RSI (CRSI) which has proven 75% win rate in literature:
1. 1d HMA(21) = MACRO BIAS (only long if price > 1d HMA, only short if price < 1d HMA)
2. 12h HMA(21) = INTERMEDIATE TREND (confirms direction, reduces whipsaws)
3. 4h CRSI(3,2,100) = ENTRY TIMING (CRSI<15 long, CRSI>85 short in range; CRSI<30/70 in trend)
4. 4h Donchian(20) = BREAKOUT CONFIRMATION (price breaks 20-bar high/low)
5. ATR(14) trailing stop at 2.5x for risk management
6. RELAXED thresholds: CRSI<25 (not <10) to ensure 30-50 trades/year

KEY INSIGHT: CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
This captures short-term oversold/overbought better than standard RSI(14).
Combined with dual HMA bias (12h + 1d), this filters counter-trend traps.

TARGET: 30-50 trades/year on 4h, Sharpe > 0.4 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_donchian_12h1d_hma_dual_bias_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    Entry: CRSI < 15 (long), CRSI > 85 (short) for mean reversion
    Entry: CRSI < 30 (long), CRSI > 70 (short) for trend pullbacks
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - short term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - consecutive up/down days
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
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rsi = 100.0 - (100.0 / (1.0 + avg_streak_gain / (avg_streak_loss + 1e-10)))
    streak_rsi = streak_rsi.fillna(50.0).values
    
    # Percent Rank - today's return vs last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period+1:i+1].dropna()
        if len(window) > 0:
            current_return = returns.iloc[i]
            percent_rank[i] = (window < current_return).sum() / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels (20-period high/low)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
    # HMA for trend detection on 4h
    hma_16_4h = calculate_hma(close, period=16)
    hma_48_4h = calculate_hma(close, period=48)
    
    # Calculate and align 12h HMA for intermediate trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro bias (HARD FILTER)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 4h (target 30-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_16_4h[i]) or np.isnan(hma_48_4h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA - HARD FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (12h HMA - CONFIRMATION) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_bullish_4h = hma_16_4h[i] > hma_48_4h[i]
        hma_bearish_4h = hma_16_4h[i] < hma_48_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # breaks previous high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # breaks previous low
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG CONDITIONS (must have 1d bullish bias)
        if price_above_hma_1d:
            # Strong long: All timeframes aligned + CRSI pullback
            if price_above_hma_12h and hma_bullish_4h:
                # Trend pullback entry (CRSI < 35, not too extreme)
                if crsi[i] < 35 and crsi[i] > 10:
                    desired_signal = BASE_SIZE
                # Donchian breakout confirmation
                elif donchian_breakout_long and crsi[i] < 50:
                    desired_signal = BASE_SIZE
            
            # Moderate long: 1d bullish + CRSI oversold (mean reversion in uptrend)
            elif crsi[i] < 20:
                desired_signal = BASE_SIZE * 0.8
        
        # SHORT CONDITIONS (must have 1d bearish bias)
        if price_below_hma_1d:
            # Strong short: All timeframes aligned + CRSI pullback
            if price_below_hma_12h and hma_bearish_4h:
                # Trend pullback entry (CRSI > 65, not too extreme)
                if crsi[i] > 65 and crsi[i] < 90:
                    desired_signal = -BASE_SIZE
                # Donchian breakout confirmation
                elif donchian_breakout_short and crsi[i] > 50:
                    desired_signal = -BASE_SIZE
            
            # Moderate short: 1d bearish + CRSI overbought (mean reversion in downtrend)
            elif crsi[i] > 80:
                desired_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 75:
            # Long position: exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25:
            # Short position: exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if bias still valid
            if position_side > 0:
                if price_above_hma_1d and crsi[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if price_below_hma_1d and crsi[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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