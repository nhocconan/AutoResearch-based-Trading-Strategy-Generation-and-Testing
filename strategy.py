#!/usr/bin/env python3
"""
Experiment #119: 4h Primary + 1d HTF — Choppiness Regime + Dual Mode Strategy

Hypothesis: Previous strategies failed because they used ONE approach (trend OR mean-revert)
for all market conditions. Research shows Choppiness Index (CHOP) effectively distinguishes
regimes: CHOP>61.8 = ranging (mean-revert works), CHOP<38.2 = trending (trend-follow works).

This strategy uses DUAL MODE:
1) CHOPPY REGIME (CHOP>55): Connors RSI mean reversion at Bollinger extremes
   - Long: CRSI<15 + price<BB_lower + 1d trend neutral/up
   - Short: CRSI>85 + price>BB_upper + 1d trend neutral/down
2) TRENDING REGIME (CHOP<45): HMA trend + Donchian breakout
   - Long: price>HMA_4h_21 + Donchian breakout + 1d HMA up
   - Short: price<HMA_4h_21 + Donchian breakdown + 1d HMA down

Why this should work:
- Adapts to market conditions instead of forcing one approach
- CHOP filter prevents trend-following in whipsaw ranges (2022 crash killer)
- CRSI mean-reversion catches oversold bounces in bear markets
- 1d HMA provides macro bias to avoid counter-trend trades
- 4h TF naturally produces 25-50 trades/year (low fee drag)

Position size: 0.25 base, 0.30 max with confluence
Stoploss: 2.5*ATR trailing
Target: Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_dual_mode_1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)  # avoid div by zero
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_percentile=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank (100)
    percent_rank = close_s.rolling(window=pr_percentile, min_periods=pr_percentile).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_percentile=100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop_14[i] > 55.0  # ranging market
        trending_regime = chop_14[i] < 45.0  # trending market
        # neutral zone: 45-55 (use previous signal or flat)
        
        # === 1d MACRO TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_1d_up = hma_1d_slope[i] > 0.3
        hma_1d_down = hma_1d_slope[i] < -0.3
        hma_1d_flat = abs(hma_1d_slope[i]) <= 0.3
        
        # === 4h TREND FILTER ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === MEAN REVERSION SIGNALS (Choppy Regime) ===
        price_at_bb_lower = close[i] <= bb_lower[i] * 1.002  # within 0.2% of lower band
        price_at_bb_upper = close[i] >= bb_upper[i] * 0.998  # within 0.2% of upper band
        
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        mr_long_signal = choppy_regime and price_at_bb_lower and crsi_oversold
        mr_short_signal = choppy_regime and price_at_bb_upper and crsi_overbought
        
        # === TREND FOLLOWING SIGNALS (Trending Regime) ===
        prev_donchian_high = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_donchian_low = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        donchian_breakout_long = close[i] > prev_donchian_high
        donchian_breakout_short = close[i] < prev_donchian_low
        
        tf_long_signal = trending_regime and hma_4h_bullish and donchian_breakout_long
        tf_short_signal = trending_regime and hma_4h_bearish and donchian_breakout_short
        
        # === ENTRY LOGIC WITH 1d BIAS FILTER ===
        new_signal = 0.0
        
        # LONG entry
        if mr_long_signal:
            # Mean reversion long: allow if 1d not strongly bearish
            if not (price_below_hma_1d and hma_1d_down):
                new_signal = POSITION_SIZE_BASE
                if hma_1d_flat or hma_1d_up:
                    new_signal = POSITION_SIZE_MAX
        
        if tf_long_signal:
            # Trend long: require 1d bias neutral or up
            if price_above_hma_1d or hma_1d_flat:
                new_signal = POSITION_SIZE_BASE
                if hma_1d_up:
                    new_signal = POSITION_SIZE_MAX
        
        # SHORT entry
        if mr_short_signal:
            # Mean reversion short: allow if 1d not strongly bullish
            if not (price_above_hma_1d and hma_1d_up):
                new_signal = -POSITION_SIZE_BASE
                if hma_1d_flat or hma_1d_down:
                    new_signal = -POSITION_SIZE_MAX
        
        if tf_short_signal:
            # Trend short: require 1d bias neutral or down
            if price_below_hma_1d or hma_1d_flat:
                new_signal = -POSITION_SIZE_BASE
                if hma_1d_down:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price above BB mid and CRSI not overbought
                if close[i] > bb_mid[i] and crsi[i] < 75.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price below BB mid and CRSI not oversold
                if close[i] < bb_mid[i] and crsi[i] > 25.0:
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
        
        # === EXIT ON CRSI EXTREME (take profit for mean reversion) ===
        if in_position and position_side > 0 and crsi[i] > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30.0:
            new_signal = 0.0
        
        # === EXIT ON REGIME CHANGE ===
        # If in trend position and regime becomes choppy, exit
        if in_position and position_side > 0 and choppy_regime:
            if tf_long_signal == False and mr_long_signal == False:
                new_signal = 0.0
        
        if in_position and position_side < 0 and choppy_regime:
            if tf_short_signal == False and mr_short_signal == False:
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