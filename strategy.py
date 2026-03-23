#!/usr/bin/env python3
"""
Experiment #165: 1h Primary + 4h/1d HTF — Regime-Adaptive Strategy with Choppiness Index

Hypothesis: Pure trend-following (#164) failed because 2022 crash and 2025 bear market
punish trend strategies with whipsaws. Pure mean-reversion also failed because it fights
strong trends. Solution: REGIME-ADAPTIVE approach that switches logic based on market state.

1) Choppiness Index (CHOP) regime detection:
   - CHOP > 55 = RANGE regime → use Connors RSI mean reversion
   - CHOP < 45 = TREND regime → use HMA trend following
   - 45-55 = NO TRADE zone (unclear regime)

2) 4h HMA(21) + 1d HMA(21) for macro bias:
   - Only take LONG if price > 4h HMA AND price > 1d HMA
   - Only take SHORT if price < 4h HMA AND price < 1d HMA
   - This prevents counter-trend trades that destroy Sharpe

3) Connors RSI for mean-reversion entries (RANGE regime):
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + regime=RANGE + HTF bullish
   - Short: CRSI > 85 + regime=RANGE + HTF bearish

4) HMA crossover for trend entries (TREND regime):
   - HMA(8) crosses above HMA(21) + regime=TREND + HTF bullish
   - HMA(8) crosses below HMA(21) + regime=TREND + HTF bearish

5) Session filter: Only trade 8-20 UTC (high liquidity, lower spread)
6) Volume filter: volume > 0.8x 20-bar average
7) ATR(14) trailing stop: 2.5x ATR for risk management
8) Position size: 0.20 base, 0.30 with full confluence (discrete levels)

Target: 40-80 trades/year on 1h, Sharpe > 0.5 on ALL symbols
Why this should work: Adapts to market regime instead of forcing one style.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_chop_crsi_hma_4h1d_v1"
timeframe = "1h"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_s = pd.Series(tr)
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum().values
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    chop[:period] = np.nan
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = min(100, streak_abs[i] * 50)
        elif streak[i] < 0:
            streak_rsi[i] = max(0, 100 - streak_abs[i] * 50)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (pr_period - 1)
    
    # CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    hma_8 = calculate_hma(close, period=8)
    hma_21 = calculate_hma(close, period=21)
    
    # Calculate 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract hour from open_time for session filter
    # open_time is in milliseconds since epoch
    hours = (open_time // 3600000) % 24
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high liquidity hours
        session_ok = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === REGIME DETECTION ===
        regime_range = chop_14[i] > 55.0  # Range/choppy market
        regime_trend = chop_14[i] < 45.0  # Trending market
        regime_neutral = not regime_range and not regime_trend  # 45-55 = no trade
        
        # === HTF TREND ALIGNMENT ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Full HTF confluence
        htf_bullish = price_above_hma_4h and price_above_hma_1d
        htf_bearish = price_below_hma_4h and price_below_hma_1d
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if not regime_neutral and session_ok and volume_ok:
            # --- RANGE REGIME: Mean Reversion with Connors RSI ---
            if regime_range:
                # Long: CRSI extremely oversold + HTF bullish
                if crsi[i] < 15.0 and htf_bullish:
                    new_signal = POSITION_SIZE_MAX
                # Short: CRSI extremely overbought + HTF bearish
                elif crsi[i] > 85.0 and htf_bearish:
                    new_signal = -POSITION_SIZE_MAX
                # Partial: CRSI extreme + one HTF aligned
                elif crsi[i] < 20.0 and price_above_hma_4h:
                    new_signal = POSITION_SIZE_BASE
                elif crsi[i] > 80.0 and price_below_hma_4h:
                    new_signal = -POSITION_SIZE_BASE
            
            # --- TREND REGIME: HMA Crossover ---
            elif regime_trend:
                # HMA crossover signals
                hma_cross_long = hma_8[i] > hma_21[i] and hma_8[i-1] <= hma_21[i-1]
                hma_cross_short = hma_8[i] < hma_21[i] and hma_8[i-1] >= hma_21[i-1]
                
                # Long: HMA cross up + HTF bullish
                if hma_cross_long and htf_bullish:
                    new_signal = POSITION_SIZE_MAX
                # Short: HMA cross down + HTF bearish
                elif hma_cross_short and htf_bearish:
                    new_signal = -POSITION_SIZE_MAX
                # Partial: HMA aligned + one HTF
                elif hma_8[i] > hma_21[i] and price_above_hma_4h:
                    new_signal = POSITION_SIZE_BASE
                elif hma_8[i] < hma_21[i] and price_below_hma_4h:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime/trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if HTF still bullish or regime supports
                if (price_above_hma_4h and (regime_trend or regime_range)) or crsi[i] < 50:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if HTF still bearish or regime supports
                if (price_below_hma_4h and (regime_trend or regime_range)) or crsi[i] > 50:
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
        # Exit if regime becomes neutral (unclear direction)
        if in_position and regime_neutral:
            new_signal = 0.0
        
        # Exit long if HTF turns bearish
        if in_position and position_side > 0 and price_below_hma_4h:
            new_signal = 0.0
        
        # Exit short if HTF turns bullish
        if in_position and position_side < 0 and price_above_hma_4h:
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