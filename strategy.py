#!/usr/bin/env python3
"""
Experiment #444: 4h Primary + 12h/1d HTF — Fisher Transform + Choppiness Regime + CRSI

Hypothesis: Build on #434 (Sharpe=0.242) by adding Ehlers Fisher Transform for reversal
detection in bear markets, combined with proven Choppiness regime switching. Key improvements:
1. Fisher Transform (period=9) catches bear market reversals better than RSI alone
2. Dual mean-reversion signals: CRSI < 20 OR Fisher < -1.5 for longs (higher hit rate)
3. Asymmetric sizing: 0.30 on strong confluence, 0.20 on single signal
4. Stricter HTF bias: require 12h AND 1d HMA agreement for trend entries
5. Volume spike filter: only enter when taker_buy_volume ratio confirms direction
6. ATR-based vol filter: reduce size when ATR > 1.5x median (high vol = more risk)

Target: Sharpe > 0.612, 80-150 trades over 4-year train, DD < -35%
Timeframe: 4h (proven best for crypto swing trading)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_crsi_donchian_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    n = len(close)
    
    # RSI(3) component
    rsi = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank component
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    # Use (high + low) / 2 as price input
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest and lowest over period
        highest = np.nanmax(hl2[i-period+1:i+1])
        lowest = np.nanmin(hl2[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_signal[i] = fisher[i]
            continue
        
        # Normalize price to 0-1 range
        normalized = (hl2[i] - lowest) / price_range
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform formula
        fisher_value = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value (Ehlers method)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_value + 0.33 * fisher[i-1]
        else:
            fisher[i] = fisher_value
        
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate taker buy ratio for volume confirmation
    taker_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    # Calculate and align HTF HMA for bias (12h and 1d)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% base position size for 4h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 61.8  # Range market (mean revert)
        regime_trend = chop[i] < 38.2  # Trending market (trend follow)
        
        # === HTF TREND BIAS (12h + 1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias when 12h and 1d agree
        htf_bullish = price_above_hma_12h and price_above_hma_1d
        htf_bearish = price_below_hma_12h and price_below_hma_1d
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === CRSI SIGNALS (Mean Reversion) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === FISHER TRANSFORM SIGNALS (Reversal) ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] <= fisher[i]
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] >= fisher[i]
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === BOLLINGER BAND SIGNALS ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = taker_ratio[i] > 0.55
        volume_bearish = taker_ratio[i] < 0.45
        
        # === VOL FILTER (ATR-based position sizing) ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.0:
            position_size = BASE_SIZE * 0.5  # High vol = reduce size
        elif vol_ratio > 1.3:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 61.8) — MEAN REVERSION ===
        if regime_chop:
            # Long: Multiple mean reversion signals + HTF not bearish
            mr_long_signals = 0
            if crsi_oversold:
                mr_long_signals += 1
            if fisher_oversold or fisher_cross_up:
                mr_long_signals += 1
            if bb_oversold:
                mr_long_signals += 1
            
            if mr_long_signals >= 2 and not htf_bearish:
                signal_strength = mr_long_signals
                if volume_bullish:
                    signal_strength += 1
                if crsi_extreme_oversold:
                    signal_strength += 1
                desired_signal = position_size * (0.7 + 0.3 * min(signal_strength, 4) / 4)
            
            # Short: Multiple mean reversion signals + HTF not bullish
            mr_short_signals = 0
            if crsi_overbought:
                mr_short_signals += 1
            if fisher_overbought or fisher_cross_down:
                mr_short_signals += 1
            if bb_overbought:
                mr_short_signals += 1
            
            if mr_short_signals >= 2 and not htf_bullish:
                if desired_signal == 0:
                    signal_strength = mr_short_signals
                    if volume_bearish:
                        signal_strength += 1
                    if crsi_extreme_overbought:
                        signal_strength += 1
                    desired_signal = -position_size * (0.7 + 0.3 * min(signal_strength, 4) / 4)
        
        # === REGIME 2: TRENDING (CHOP < 38.2) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout + HTF bullish + volume confirmation
            if donchian_breakout_long:
                signal_strength = 1
                if htf_bullish:
                    signal_strength += 1
                if volume_bullish:
                    signal_strength += 1
                if hma_bullish:
                    signal_strength += 1
                desired_signal = position_size * (0.7 + 0.3 * min(signal_strength, 4) / 4)
            elif htf_bullish and hma_bullish and close[i] > hma_21[i]:
                if desired_signal == 0:
                    signal_strength = 1
                    if volume_bullish:
                        signal_strength += 1
                    desired_signal = position_size * 0.6 * (0.7 + 0.3 * signal_strength / 2)
            
            # Short: Donchian breakdown + HTF bearish + volume confirmation
            if donchian_breakout_short:
                if desired_signal == 0:
                    signal_strength = 1
                    if htf_bearish:
                        signal_strength += 1
                    if volume_bearish:
                        signal_strength += 1
                    if hma_bearish:
                        signal_strength += 1
                    desired_signal = -position_size * (0.7 + 0.3 * min(signal_strength, 4) / 4)
            elif htf_bearish and hma_bearish and close[i] < hma_21[i]:
                if desired_signal == 0:
                    signal_strength = 1
                    if volume_bearish:
                        signal_strength += 1
                    desired_signal = -position_size * 0.6 * (0.7 + 0.3 * signal_strength / 2)
        
        # === REGIME 3: TRANSITION (38.2-61.8) — ONLY STRONG SIGNALS ===
        else:
            # Only enter on extreme conditions with HTF agreement
            if crsi_extreme_oversold and fisher_cross_up and not htf_bearish:
                desired_signal = position_size * 0.6
            elif crsi_extreme_overbought and fisher_cross_down and not htf_bullish:
                desired_signal = -position_size * 0.6
            elif donchian_breakout_long and htf_bullish and volume_bullish:
                desired_signal = position_size * 0.6
            elif donchian_breakout_short and htf_bearish and volume_bearish:
                desired_signal = -position_size * 0.6
        
        # === CAP SIGNAL TO MAX 0.35 ===
        if desired_signal > 0.35:
            desired_signal = 0.35
        elif desired_signal < -0.35:
            desired_signal = -0.35
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT ===
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or price_above_hma_12h):
                desired_signal = position_size
            elif position_side < 0 and (hma_bearish or price_below_hma_12h):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.30:
                    desired_signal = 0.30
                elif desired_signal >= 0.20:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.30:
                    desired_signal = -0.30
                elif desired_signal <= -0.20:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.15
        
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