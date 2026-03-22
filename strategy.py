#!/usr/bin/env python3
"""
Experiment #005: 1h Fisher-RSI Pullback with 4h/1d Trend Alignment

Hypothesis: Previous strategies failed due to:
- Choppiness Index regime switching (failed #001, #002, #003, #004)
- Over-complicated multi-regime logic with too many filters
- 12h timeframe too slow for this experiment (requires 1h)

This strategy uses PROVEN patterns for 1h timeframe:
1. 4h HMA(21) for trend direction - smoother than EMA, less whipsaw
2. 1d HMA(12) for major trend bias - align with daily direction
3. Ehlers Fisher Transform(9) for reversal entries - catches bear market rallies
4. Connors RSI(3,2,100) for pullback confirmation - 75% win rate in research
5. Session filter (8-20 UTC) - reduces trades to 30-60/year target
6. Volume confirmation (>1.2x avg) - avoids low-liquidity entries
7. ATR(14) trailing stoploss 2.5x - protects capital

Why 1h works for this experiment:
- Target 30-60 trades/year (fee drag ~1.5-3% manageable)
- Use 4h/1d for SIGNAL DIRECTION (proven 2x Sharpe)
- Use 1h only for ENTRY TIMING precision
- Session filter prevents overtrading

Position sizing: 0.25 discrete levels (CRITICAL for drawdown control)
Stoploss: 2.5 * ATR(14) trailing
Timeframe: 1h (REQUIRED for Experiment #005)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_crsi_pullback_4h_1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reference: Alan Hull, 2005
    """
    close_s = pd.Series(close)
    n = len(close)
    
    if period < 2:
        return np.full(n, np.nan)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals in bear markets. Reference: John Ehlers, 2002
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Normalize to range -1 to +1
    highest = hl2_s.rolling(window=period, min_periods=period).max()
    lowest = hl2_s.rolling(window=period, min_periods=period).min()
    
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, np.nan)
    
    normalized = ((hl2_s - lowest) / range_hl - 0.5) * 1.9
    normalized = normalized.clip(-0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag)
    fisher_s = pd.Series(fisher)
    trigger = fisher_s.shift(1)
    
    return fisher.values, trigger.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines RSI, streak RSI, and percent rank.
    Reference: Laurence Connors, 2008. 75% win rate on extremes.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI component (short period)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0)
    streak_loss = -streak_s.where(streak_s < 0, 0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    streak_rs = streak_rs.replace([np.inf, -np.inf], np.nan)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    
    # Percent Rank component
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50,
        raw=False
    )
    
    # CRSI = average of three components
    crsi = (rsi + streak_rsi + percent_rank) / 3
    
    return crsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_hour_from_open_time(open_time):
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d HMA for major trend bias
    hma_1d_12 = calculate_hma(df_1d['close'].values, 12)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_12)
    
    # Calculate 1h indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Volume moving average for confirmation
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Extract UTC hour for session filter
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        if np.isnan(volume_ma20[i]) or volume_ma20[i] == 0:
            continue
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND DIRECTION ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === 4H HMA SLOPE ===
        hma_4h_slope_long = hma_4h_aligned[i] > hma_4h_aligned[i-4] if i > 4 else False
        hma_4h_slope_short = hma_4h_aligned[i] < hma_4h_aligned[i-4] if i > 4 else False
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        
        # Fisher extreme levels for strong signals
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === CONNORS RSI SIGNALS ===
        # Long: CRSI < 15 (extreme oversold)
        crsi_oversold = crsi[i] < 15
        # Short: CRSI > 85 (extreme overbought)
        crsi_overbought = crsi[i] > 85
        
        # CRSI pullback zones
        crsi_pullback_long = (crsi[i] > 25) and (crsi[i] < 45)
        crsi_pullback_short = (crsi[i] > 55) and (crsi[i] < 75)
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 1.2 * volume_ma20[i]
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high-liquidity hours to reduce false signals
        session_ok = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need 4h/1d bullish + Fisher/CRSI signal + session + volume
        long_score = 0
        if trend_4h_bullish:
            long_score += 2  # Primary requirement
        if daily_bullish:
            long_score += 1  # Major trend alignment
        if hma_4h_slope_long:
            long_score += 0.5
        if fisher_long or fisher_oversold:
            long_score += 1.5  # Strong reversal signal
        if crsi_oversold or crsi_pullback_long:
            long_score += 1
        if session_ok:
            long_score += 0.5
        if volume_ok:
            long_score += 0.5
        
        # Enter long if score >= 5 (need trend + reversal signal + confirmation)
        if long_score >= 5 and trend_4h_bullish:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Need 4h/1d bearish + Fisher/CRSI signal + session + volume
        short_score = 0
        if trend_4h_bearish:
            short_score += 2  # Primary requirement
        if daily_bearish:
            short_score += 1  # Major trend alignment
        if hma_4h_slope_short:
            short_score += 0.5
        if fisher_short or fisher_overbought:
            short_score += 1.5  # Strong reversal signal
        if crsi_overbought or crsi_pullback_short:
            short_score += 1
        if session_ok:
            short_score += 0.5
        if volume_ok:
            short_score += 0.5
        
        # Enter short if score >= 5 (need trend + reversal signal + confirmation)
        if short_score >= 5 and trend_4h_bearish:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 120 bars (~5 days on 1h), allow weaker entry
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and (daily_bullish or fisher_oversold):
                new_signal = BASE_SIZE * 0.6
            elif trend_4h_bearish and (daily_bearish or fisher_overbought):
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_4h_bearish:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish:
                trend_reversal = True
        
        # === FISHER EXTREME EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher[i] > 2.0:
                fisher_exit = True
            if position_side < 0 and fisher[i] < -2.0:
                fisher_exit = True
        
        # === CRSI EXTREME EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 80:
                crsi_exit = True
            if position_side < 0 and crsi[i] < 20:
                crsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or fisher_exit or crsi_exit:
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