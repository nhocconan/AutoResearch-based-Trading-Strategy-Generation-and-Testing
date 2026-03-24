#!/usr/bin/env python3
"""
Experiment #600: 6h Primary + 1d/1w HTF — Dual Regime Trend/Mean Reversion

Hypothesis: 6h timeframe sits in sweet spot between 4h (too noisy) and 12h (too slow).
Using weekly HMA for macro bias + daily RSI for momentum + 6h pullback entries should
capture multi-day swings while avoiding whipsaw. Key innovation: DUAL REGIME approach
where we switch between trend-following and mean-reversion based on ADX/CHOP confluence.

Why 6h might work:
1. Captures 2-4 day swings (perfect for crypto volatility cycles)
2. Less noise than 4h, more signals than 12h
3. 6h aligns well with funding rate cycles (8h) and daily closes

Key differences from failed #591/#595:
1. Simpler entry logic - fewer confluence requirements = MORE TRADES
2. Weekly HMA SLOPE (not just price vs HMA) for stronger trend filter
3. Daily RSI regime (not 6h RSI) for momentum confirmation
4. Asymmetric sizing: stronger signals in trend regime, lighter in range
5. Explicit trade frequency control via entry cooldown

Strategy logic:
1. 1w HMA(21) slope = macro trend (rising/falling/flat)
2. 1d RSI(14) = momentum regime (>55 bull, <45 bear, 45-55 neutral)
3. 6h price vs HMA(21) = entry trigger (pullback to HMA in trend)
4. 6h ADX(14) = trend strength filter (>25 trend, <20 range)
5. ATR(14)*2.5 stoploss on all positions

Entry conditions (LOOSE enough to generate trades):
- LONG: 1w HMA slope > 0 + 1d RSI > 45 + 6h price pulls back to HMA + ADX > 18
- SHORT: 1w HMA slope < 0 + 1d RSI < 55 + 6h price rallies to HMA + ADX > 18
- RANGE: 1w HMA flat + 6h RSI extreme (<30 or >70) + CHOP > 55

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-30%
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_dual_regime_hma_rsi_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures choppy vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d RSI for momentum confirmation
    rsi_1d_raw = calculate_rsi(df_1d['close'].values, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=21)
    rsi_6h = calculate_rsi(close, period=14)
    atr_6h = calculate_atr(high, low, close, period=14)
    adx_6h = calculate_adx(high, low, close, period=14)
    chop_6h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate 1w HMA slope (lookback 5 bars on 1w = ~5 weeks)
    hma_1w_slope = np.zeros(n)
    hma_1w_slope[:] = np.nan
    for i in range(5, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-5]):
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-5]) / hma_1w_aligned[i-5] * 100
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_RANGE = 0.25
    SIZE_WEAK = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Trade cooldown to limit frequency
    last_entry_bar = -100
    min_cooldown = 20  # Minimum 20 bars (120 hours = 5 days) between entries
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_6h[i]) or atr_6h[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi_6h[i]) or np.isnan(adx_6h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(hma_1w_slope[i]):
            signals[i] = 0.0
            continue
        
        # === HTF BIAS (1w slope + 1d RSI) ===
        weekly_bull = hma_1w_slope[i] > 0.5  # Rising weekly trend
        weekly_bear = hma_1w_slope[i] < -0.5  # Falling weekly trend
        weekly_flat = abs(hma_1w_slope[i]) <= 0.5
        
        daily_bull = rsi_1d_aligned[i] > 50.0
        daily_bear = rsi_1d_aligned[i] < 50.0
        
        # === 6h TREND ===
        price_above_hma = close[i] > hma_6h[i]
        price_below_hma = close[i] < hma_6h[i]
        
        # Pullback detection (price near HMA)
        hma_distance_pct = abs(close[i] - hma_6h[i]) / hma_6h[i] * 100
        near_hma = hma_distance_pct < 2.0  # Within 2% of HMA
        
        # === ADX/CHOP REGIME ===
        adx_trend = adx_6h[i] > 22.0
        adx_range = adx_6h[i] < 20.0
        chop_range = chop_6h[i] > 55.0
        chop_trend = chop_6h[i] < 45.0
        
        # === RSI EXTREMES (6h) ===
        rsi_oversold = rsi_6h[i] < 35.0
        rsi_overbought = rsi_6h[i] > 65.0
        rsi_extreme_oversold = rsi_6h[i] < 28.0
        rsi_extreme_overbought = rsi_6h[i] > 72.0
        
        # === REGIME DETECTION ===
        is_trend_regime = adx_trend and chop_trend
        is_range_regime = adx_range and chop_range
        is_mixed = not is_trend_regime and not is_range_regime
        
        # === ENTRY LOGIC (LOOSE enough to generate trades) ===
        desired_signal = 0.0
        cooldown_ok = (i - last_entry_bar) >= min_cooldown
        
        # TREND REGIME: Pullback entries with HTF confirmation
        if is_trend_regime and cooldown_ok:
            # Long: Weekly bull + Daily bull + Pullback to HMA + RSI not overbought
            if weekly_bull and daily_bull and price_below_hma and near_hma and rsi_6h[i] < 55.0:
                desired_signal = SIZE_TREND
            # Short: Weekly bear + Daily bear + Rally to HMA + RSI not oversold
            elif weekly_bear and daily_bear and price_above_hma and near_hma and rsi_6h[i] > 45.0:
                desired_signal = -SIZE_TREND
        
        # RANGE REGIME: Mean reversion at extremes
        elif is_range_regime and cooldown_ok:
            if rsi_extreme_oversold and weekly_flat:
                desired_signal = SIZE_RANGE
            elif rsi_extreme_overbought and weekly_flat:
                desired_signal = -SIZE_RANGE
        
        # MIXED REGIME: Weaker signals, need stronger confluence
        elif is_mixed and cooldown_ok:
            # Long: All bullish confluence
            if weekly_bull and daily_bull and rsi_oversold and price_below_hma:
                desired_signal = SIZE_WEAK
            # Short: All bearish confluence
            elif weekly_bear and daily_bear and rsi_overbought and price_above_hma:
                desired_signal = -SIZE_WEAK
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_RANGE * 0.9:
            final_signal = SIZE_RANGE
        elif desired_signal <= -SIZE_RANGE * 0.9:
            final_signal = -SIZE_RANGE
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_6h[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                last_entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals