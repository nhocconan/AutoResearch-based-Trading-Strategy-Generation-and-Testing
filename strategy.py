#!/usr/bin/env python3
"""
Experiment #1535: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 1140+ failed strategies, the pattern for 1h timeframe is:
1. Connors RSI too strict → 0 trades (#1528, #1529, #1530 all failed with Sharpe=0)
2. Session filters kill trade frequency on lower TF
3. 4h HMA trend bias works better than 12h/1d for 1h entries (more responsive)
4. Ehlers Fisher Transform proven for bear/range markets (research notes)
5. Softer Choppiness thresholds (>50 range, <40 trend) generate more signals
6. Volume confirmation simple: >0.8x 20-bar average
7. RSI(7) faster than RSI(14) for 1h entries

Design:
- Primary: 1h timeframe (as required by experiment #1535)
- HTF: 4h HMA(21) for trend bias, 1d HMA(21) for macro filter
- Regime: Choppiness(14) — range when >50, trend when <40
- Entry trigger: Fisher Transform cross -1.5 (long) / +1.5 (short)
- Momentum filter: RSI(7) not extreme against position
- Volume filter: volume > 0.8x 20-bar average
- Stoploss: ATR(14) 2.5x trailing
- Position size: 0.25 (smaller for 1h to reduce fee drag)
- Target: 40-60 trades/train, 10-15 trades/test

Why this should work:
- 1h TF = more opportunities than 4h while using 4h for direction
- Fisher Transform catches reversals in bear rallies (research proven)
- Softer CHOP thresholds ensure trades fire (learned from 0-trade failures)
- 4h HMA bias prevents counter-trend trades without being too slow
- Volume filter adds confluence without being too restrictive
- Discrete sizing (0.0, ±0.25) minimizes fee churn

Timeframe: 1h (as required by experiment #1535)
HTF: 4h (trend bias), 1d (macro filter)
Position Size: 0.25 (conservative for 1h volatility)
Target: Sharpe > 0.618 (beat current best), DD < -30%, trades > 30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_hma_4h1d_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 50 = range/choppy market (softer threshold for more signals)
    CHOP < 40 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points in bear/range markets
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.67
    Signal line = Fisher shifted by 1
    Long: Fisher crosses above -1.5
    Short: Fisher crosses below +1.5
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    fisher = np.full(n, np.nan)
    signal = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            X = 0.67 * (typical[i] - lowest) / price_range - 0.67
            X = np.clip(X, -0.99, 0.99)  # prevent division by zero
            fisher[i] = 0.5 * np.log((1.0 + X) / (1.0 - X))
        
        if i > period and not np.isnan(fisher[i-1]):
            signal[i] = fisher[i-1]
    
    return fisher, signal

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 40.0
        
        # === MACRO TREND BIAS (4h HMA) ===
        fourh_bull = close[i] > hma_4h_aligned[i]
        fourh_bear = close[i] < hma_4h_aligned[i]
        
        # === DAILY FILTER (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # === RSI MOMENTUM FILTER ===
        rsi_not_overbought = rsi_7[i] < 70.0
        rsi_not_oversold = rsi_7[i] > 30.0
        
        # === DESIRED SIGNAL — REGIME-ADAPTIVE LOGIC ===
        desired_signal = 0.0
        
        if is_choppy:
            # === RANGE REGIME: Fisher Mean Reversion ===
            if fisher_extreme_low and fourh_bull and volume_ok and rsi_not_overbought:
                # Strong long: Fisher extreme + 4h bull + volume
                desired_signal = BASE_SIZE
            elif fisher_extreme_high and fourh_bear and volume_ok and rsi_not_oversold:
                # Strong short: Fisher extreme + 4h bear + volume
                desired_signal = -BASE_SIZE
            elif fisher_cross_up and fourh_bull and volume_ok:
                # Fisher cross up in uptrend
                desired_signal = BASE_SIZE * 0.8
            elif fisher_cross_down and fourh_bear and volume_ok:
                # Fisher cross down in downtrend
                desired_signal = -BASE_SIZE * 0.8
            elif fisher_cross_up and daily_bull:
                # Daily bull + Fisher cross (weaker signal)
                desired_signal = BASE_SIZE * 0.6
            elif fisher_cross_down and daily_bear:
                # Daily bear + Fisher cross (weaker signal)
                desired_signal = -BASE_SIZE * 0.6
        else:
            # === TREND REGIME: Follow 4h HMA direction ===
            if fourh_bull and daily_bull:
                # Strong long bias (both 4h and 1d bullish)
                if fisher_cross_up and volume_ok:
                    desired_signal = BASE_SIZE
                elif fisher[i] > -1.0 and rsi_not_overbought:
                    desired_signal = BASE_SIZE * 0.7
            elif fourh_bear and daily_bear:
                # Strong short bias (both 4h and 1d bearish)
                if fisher_cross_down and volume_ok:
                    desired_signal = -BASE_SIZE
                elif fisher[i] < 1.0 and rsi_not_oversold:
                    desired_signal = -BASE_SIZE * 0.7
            elif fourh_bull:
                # 4h bullish only
                if fisher_cross_up and volume_ok:
                    desired_signal = BASE_SIZE * 0.7
            elif fourh_bear:
                # 4h bearish only
                if fisher_cross_down and volume_ok:
                    desired_signal = -BASE_SIZE * 0.7
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals