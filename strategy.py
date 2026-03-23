#!/usr/bin/env python3
"""
Experiment #110: 1h Primary + 4h/12h HTF — Regime-Adaptive Mean Reversion

Hypothesis: Pure trend-following fails in bear/range markets (2022 crash, 2025 bear).
This strategy uses regime detection to switch between mean reversion (range) and
trend pullback (trending) entries, with HTF trend filter for bias.

Key innovations:
1) 4h HMA(21) for macro trend bias — only trade WITH HTF trend direction
2) 1h Choppiness Index(14) for regime — CHOP>55=range (mean revert), CHOP<45=trend (pullback)
3) 1h Connors RSI for entry timing — different thresholds per regime
4) Volume confirmation (>0.8x 20-bar avg) — filters false signals
5) Session filter (8-20 UTC) — trade during high liquidity hours
6) ATR(14) trailing stop at 2.5x — limits drawdown

Why this should work:
- Range regime (60% of time): CRSI mean reversion has 75% win rate
- Trend regime (40% of time): Pullback entries with HTF bias reduce whipsaws
- 1h timeframe: 40-60 trades/year target (low fee drag)
- Session filter: Reduces noise from Asian session low liquidity

Position size: 0.25 base, 0.30 max with volume confluence
Stoploss: 2.5*ATR trailing
Target: 40-60 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_chop_session_4h_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)  # avoid div by zero
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10-15, Short: CRSI > 85-90
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI(2)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    streak_avg_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank(100)
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    ).values
    
    crsi = (rsi.values + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        chop_value = chop_14[i]
        is_range_regime = chop_value > 55  # Mean reversion regime
        is_trend_regime = chop_value < 45  # Trend follow regime
        is_neutral_regime = not is_range_regime and not is_trend_regime
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_ok = volume_ratio > 0.8
        volume_strong = volume_ratio > 1.5
        
        # === CRSI VALUES ===
        crsi_value = crsi[i]
        crsi_oversold = crsi_value < 15  # Strong mean reversion long
        crsi_overbought = crsi_value > 85  # Strong mean reversion short
        crsi_pullback_long = crsi_value < 35  # Pullback long in trend
        crsi_pullback_short = crsi_value > 65  # Pullback short in trend
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        # --- LONG ENTRY ---
        long_signal = False
        
        # Range regime: Mean reversion (CRSI < 15)
        if is_range_regime:
            if crsi_oversold and volume_ok:
                long_signal = True
        
        # Trend regime: Pullback entry with HTF bias (CRSI < 35 + price > 4h HMA)
        elif is_trend_regime:
            if crsi_pullback_long and price_above_hma_4h and volume_ok:
                long_signal = True
        
        # Neutral regime: Conservative mean reversion with HTF bias
        elif is_neutral_regime:
            if crsi_oversold and price_above_hma_4h and volume_ok:
                long_signal = True
        
        if long_signal:
            new_signal = POSITION_SIZE_BASE
            if volume_strong:
                new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        short_signal = False
        
        # Range regime: Mean reversion (CRSI > 85)
        if is_range_regime:
            if crsi_overbought and volume_ok:
                short_signal = True
        
        # Trend regime: Pullback entry with HTF bias (CRSI > 65 + price < 4h HMA)
        elif is_trend_regime:
            if crsi_pullback_short and price_below_hma_4h and volume_ok:
                short_signal = True
        
        # Neutral regime: Conservative mean reversion with HTF bias
        elif is_neutral_regime:
            if crsi_overbought and price_below_hma_4h and volume_ok:
                short_signal = True
        
        if short_signal:
            new_signal = -POSITION_SIZE_BASE
            if volume_strong:
                new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold position if no new signal and not stopped out
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought and HTF trend intact
                if crsi_value < 80 and price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold and HTF trend intact
                if crsi_value > 20 and price_below_hma_4h:
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
        # Exit long if regime shifts to strong trend down
        if in_position and position_side > 0:
            if is_trend_regime and price_below_hma_4h:
                new_signal = 0.0
        
        # Exit short if regime shifts to strong trend up
        if in_position and position_side < 0:
            if is_trend_regime and price_above_hma_4h:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi_value > 80:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi_value < 20:
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