#!/usr/bin/env python3
"""
Experiment #604: 12h Primary + 1d/1w HTF — Donchian Breakout + HMA Trend + Choppiness Regime

Hypothesis: 12h timeframe with Donchian Channel breakouts provides cleaner trend entries
than EMA/HMA crossovers. Combined with Choppiness Index for regime detection, this
switches between trend-following (breakouts) and mean-reversion (RSI extremes) based
on market state. 12h naturally limits trades to 20-50/year, reducing fee drag.

Key innovations vs failed experiments:
1. Donchian(20) breakout instead of simple MA crossover — cleaner signals
2. Choppiness Index regime switch — trend vs range logic
3. 1d HMA + 1w HMA for HTF bias confirmation (both must align)
4. RSI(14) for entry timing within regime
5. ATR(14)*2.5 trailing stoploss on all positions
6. Discrete position sizing: 0.0, ±0.25, ±0.30

Strategy logic:
1. 1w HMA(21) = macro trend bias
2. 1d HMA(21) = medium trend bias  
3. 12h Donchian(20) = breakout detection
4. 12h HMA(21) = trend confirmation
5. 12h Choppiness(14) = regime (CHOP>61.8=range, CHOP<38.2=trend)
6. 12h RSI(14) = entry timing
7. 12h ATR(14) = stoploss and volatility filter

Regime-adaptive entries:
- TREND (CHOP<45 + ADX>25): Donchian breakout + HMA alignment + HTF confirmation
- RANGE (CHOP>55 + ADX<20): RSI mean reversion at extremes (RSI<30 long, RSI>70 short)
- TRANSITION: Stay flat or reduce size

Target: Sharpe>0.40, trades>=40 train (10/year), trades>=5 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_chop_regime_1d1w_v1"
timeframe = "12h"
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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout detection
    Upper = highest high over period
    Lower = lowest low over period
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Donchian breakout for entry confirmation
    prev_donchian_upper = 0.0
    prev_donchian_lower = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(adx[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # HMA slope (5-bar lookback)
        hma_slope_bull = hma_12h[i] > hma_12h[i-5] if i >= 5 and not np.isnan(hma_12h[i-5]) else False
        hma_slope_bear = hma_12h[i] < hma_12h[i-5] if i >= 5 and not np.isnan(hma_12h[i-5]) else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] and (i < 2 or close[i-1] <= donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False)
        donchian_breakout_short = close[i] < donchian_lower[i] and (i < 2 or close[i-1] >= donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False)
        
        # Donchian position within channel
        donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
        donchian_upper_half = close[i] > donchian_mid
        donchian_lower_half = close[i] < donchian_mid
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 25.0
        adx_weak = adx[i] < 20.0
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0
        chop_trend = chop[i] < 45.0
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_extreme_oversold = rsi[i] < 30.0
        rsi_extreme_overbought = rsi[i] > 70.0
        rsi_neutral = 40.0 <= rsi[i] <= 60.0
        
        # === REGIME DETECTION ===
        is_trend_regime = adx_strong and chop_trend
        is_range_regime = adx_weak and chop_range
        is_transition = not is_trend_regime and not is_range_regime
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Donchian breakout + HMA + HTF alignment
        if is_trend_regime:
            # Long: breakout + bullish HMA + bullish HTF
            if donchian_breakout_long and hma_bull and hma_slope_bull and htf_bull:
                desired_signal = SIZE_STRONG
            # Short: breakout + bearish HMA + bearish HTF
            elif donchian_breakout_short and hma_bear and hma_slope_bear and htf_bear:
                desired_signal = -SIZE_STRONG
            # Pullback entry in trend (RSI neutral + HMA aligned)
            elif htf_bull and hma_bull and rsi_neutral and donchian_upper_half:
                desired_signal = SIZE_BASE
            elif htf_bear and hma_bear and rsi_neutral and donchian_lower_half:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: RSI mean reversion at extremes
        elif is_range_regime:
            if rsi_extreme_oversold and hma_bull:
                desired_signal = SIZE_BASE
            elif rsi_extreme_overbought and hma_bear:
                desired_signal = -SIZE_BASE
            # RSI recovery from extreme
            elif rsi_oversold and i > 0 and rsi[i] > rsi[i-1] and not np.isnan(rsi[i-1]):
                desired_signal = SIZE_BASE * 0.8
            elif rsi_overbought and i > 0 and rsi[i] < rsi[i-1] and not np.isnan(rsi[i-1]):
                desired_signal = -SIZE_BASE * 0.8
        
        # TRANSITION REGIME: Reduced size, wait for HTF confirmation
        elif is_transition:
            if htf_bull and hma_bull and rsi_oversold:
                desired_signal = SIZE_BASE * 0.6
            elif htf_bear and hma_bear and rsi_overbought:
                desired_signal = -SIZE_BASE * 0.6
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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