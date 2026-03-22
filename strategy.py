#!/usr/bin/env python3
"""
Experiment #257: 1d Primary + 1w HTF — Dual Regime (KAMA Trend + Connors RSI Mean-Revert)

Hypothesis: After 256 experiments, the winning pattern combines:
1. 1w KAMA slope for MAJOR trend bias (bull/bear regime)
2. 1d Choppiness Index for LOCAL regime (trend vs range mode)
3. Connors RSI for mean-reversion entries in choppy markets (75% win rate)
4. KAMA crossover for trend entries in trending markets
5. Volume filter to avoid low-liquidity false signals
6. Asymmetric positioning: reduce size in uncertain regimes

Key improvements over #256 (Sharpe=0.205):
- 1w HTF for stronger trend bias (not 1d)
- Connors RSI instead of regular RSI (faster mean-revert signals)
- Volume confirmation filter (avoid low-volume traps)
- Dual regime: trend-follow OR mean-revert based on Choppiness
- Conservative sizing: 0.25 base, 0.35 strong conviction

Position sizing: 0.25 base, 0.35 strong (discrete levels)
Target: 25-50 trades/year per symbol
Stoploss: 3.0 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_kama_crsi_vol_1w_v1"
timeframe = "1d"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i >= er_period:
            signal = abs(close[i] - close[i - er_period])
            noise = 0.0
            for j in range(i - er_period + 1, i + 1):
                noise += abs(close[j] - close[j - 1])
            er = signal / noise if noise > 0 else 0.0
        else:
            er = 0.0
        
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_kama_slope(kama_values, lookback=5):
    """Calculate KAMA slope as percentage change over lookback."""
    slope = np.zeros(len(kama_values))
    for i in range(lookback, len(kama_values)):
        prev = kama_values[i - lookback]
        curr = kama_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    rsi_3 = calculate_rsi(close, rsi_period)
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, streak_period)
    
    pct_rank = np.zeros(n)
    returns = close_s.pct_change().values
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current_ret = returns[i]
            if not np.isnan(current_ret):
                pct_rank[i] = np.sum(valid < current_ret) / len(valid) * 100
    
    crsi = (rsi_3 + streak_rsi + pct_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_volume_ma(volume, period=20):
    """Calculate moving average of volume."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend bias)
    kama_1w_21 = calculate_kama(df_1w['close'].values, 10, 2, 30)
    kama_1w_slope = calculate_kama_slope(kama_1w_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1w_21_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_21)
    kama_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    
    # 1d KAMA for local trend
    kama_1d_21 = calculate_kama(close, 10, 2, 30)
    kama_1d_50 = calculate_kama(close, 10, 2, 50)
    kama_1d_slope = calculate_kama_slope(kama_1d_21, 5)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1w_21_aligned[i]) or np.isnan(kama_1w_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(kama_1d_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # === 1W MAJOR TREND BIAS ===
        # Bull: 1w KAMA slope > 0.15%
        # Bear: 1w KAMA slope < -0.15%
        major_bull = kama_1w_slope_aligned[i] > 0.15
        major_bear = kama_1w_slope_aligned[i] < -0.15
        major_neutral = not major_bull and not major_bear
        
        price_above_1w_kama = close[i] > kama_1w_21_aligned[i]
        price_below_1w_kama = close[i] < kama_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME (1d) ===
        # CHOP > 58 = range market (mean revert entries)
        # CHOP < 42 = trend market (breakout entries)
        is_choppy = chop_14[i] > 58.0
        is_trending = chop_14[i] < 42.0
        
        # === VOLUME FILTER ===
        # Only trade when volume > 0.8 * 20-day MA (avoid low-liquidity)
        volume_ok = volume[i] > 0.8 * vol_ma_20[i]
        
        # === 1D LOCAL SIGNALS ===
        price_above_1d_kama = close[i] > kama_1d_21[i]
        price_below_1d_kama = close[i] < kama_1d_21[i]
        kama_1d_bullish = kama_1d_21[i] > kama_1d_50[i]
        kama_1d_bearish = kama_1d_21[i] < kama_1d_50[i]
        kama_1d_slope_positive = kama_1d_slope[i] > 0.10
        kama_1d_slope_negative = kama_1d_slope[i] < -0.10
        
        # === CONNORS RSI THRESHOLDS ===
        crsi_oversold = crsi[i] < 18.0
        crsi_overbought = crsi[i] > 82.0
        crsi_extreme_oversold = crsi[i] < 12.0
        crsi_extreme_overbought = crsi[i] > 88.0
        crsi_mid_bull = crsi[i] > 45.0
        crsi_mid_bear = crsi[i] < 55.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + volume ok)
        if is_trending and volume_ok:
            # LONG: Major bull + price above 1w KAMA + 1d KAMA bullish + CRSI confirming
            if major_bull and price_above_1w_kama and kama_1d_bullish and crsi_mid_bull:
                new_signal = STRONG_SIZE
            # LONG: Major bull + 1d KAMA slope positive + price above 1d KAMA
            elif major_bull and kama_1d_slope_positive and price_above_1d_kama and crsi[i] > 40:
                new_signal = BASE_SIZE
            
            # SHORT: Major bear + price below 1w KAMA + 1d KAMA bearish + CRSI confirming
            if major_bear and price_below_1w_kama and kama_1d_bearish and crsi_mid_bear:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Major bear + 1d KAMA slope negative + price below 1d KAMA
            elif major_bear and kama_1d_slope_negative and price_below_1d_kama and crsi[i] < 60:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy + volume ok)
        if is_choppy and volume_ok:
            # LONG: Choppy + CRSI oversold (<18) + not in major bear
            if crsi_oversold and not major_bear:
                new_signal = BASE_SIZE
            # LONG: Choppy + CRSI extreme oversold (<12) in any regime
            if crsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.9
            
            # SHORT: Choppy + CRSI overbought (>82) + not in major bull
            if crsi_overbought and not major_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + CRSI extreme overbought (>88) in any regime
            if crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.9
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 25 bars (~25 days on 1d)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position and volume_ok:
            if major_bull and crsi[i] > 40 and price_above_1d_kama:
                new_signal = BASE_SIZE * 0.7
            elif major_bear and crsi[i] < 60 and price_below_1d_kama:
                new_signal = -BASE_SIZE * 0.7
            elif is_choppy and crsi[i] < 25:
                new_signal = BASE_SIZE * 0.6
            elif is_choppy and crsi[i] > 75:
                new_signal = -BASE_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but major trend turns strongly bearish
            if position_side > 0 and major_bear and price_below_1w_kama:
                regime_reversal = True
            # Short position but major trend turns strongly bullish
            if position_side < 0 and major_bull and price_above_1w_kama:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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