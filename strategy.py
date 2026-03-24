#!/usr/bin/env python3
"""
Experiment #587: 6h Primary + 1d/1w HTF — Fisher Transform + Regime Adaptive

Hypothesis: 6h timeframe with Ehlers Fisher Transform provides superior reversal
detection vs RSI during volatile periods. Fisher normalizes price to Gaussian
distribution, making extremes clearer. Combined with Choppiness regime detection
and dual HTF (1d/1w) bias filters, this should outperform failed HMA/RSI strategies.

Key differences from failed #580 (6h_hma_rsi_pullback):
1. Fisher Transform instead of RSI - better reversal detection in bear markets
2. Regime-adaptive entries (trend vs range) - not one-size-fits-all
3. Asymmetric entry thresholds - easier to enter with HTF bias
4. Volume confirmation on breakouts - reduces false signals
5. Looser entry conditions to ensure trade generation (learned from 0-trade failures)

Strategy logic:
1. 1w HMA(21) = macro trend bias
2. 1d HMA(21) = medium trend bias  
3. 6h Fisher(9) = entry timing (crosses ±1.5 = signal)
4. 6h Choppiness(14) = regime (CHOP>61.8 = range, CHOP<38.2 = trend)
5. 6h ADX(14) = trend strength confirmation
6. 6h Volume SMA ratio = breakout confirmation (>1.5 = valid)
7. ATR(14)*2.5 stoploss on all positions

Regime-adaptive entries:
- TREND (ADX>25 + CHOP<45): Fisher reversal + HTF alignment
- RANGE (ADX<20 + CHOP>55): Fisher extremes only (±2.0)
- TRANSITION: Reduced size, require volume confirmation

Target: Sharpe>0.40, trades>=120 train (30/year), trades>=15 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_regime_chop_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better reversal detection than RSI in range/bear markets
    
    Steps:
    1. Calculate median price = (high + low) / 2
    2. Normalize: x = (median - LL) / (HH - LL) where LL/HH are Donchian extremes
    3. Transform: Fisher = 0.5 * ln((1+x)/(1-x))
    4. Signal line = 1-period EMA of Fisher
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    signal = np.zeros(n)
    signal[:] = np.nan
    
    # Median price
    median = (high + low) / 2.0
    
    for i in range(period, n):
        # Donchian extremes over lookback
        hh = np.nanmax(high[i-period+1:i+1])
        ll = np.nanmin(low[i-period+1:i+1])
        
        price_range = hh - ll
        if price_range < 1e-10:
            fisher[i] = 0.0
            continue
        
        # Normalize to 0-1 range with bounds
        x = (median[i] - ll) / price_range
        x = max(0.001, min(0.999, x))  # Clamp to avoid ln(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
    
    # Signal line (1-period EMA = just smooth slightly)
    fisher_series = pd.Series(fisher)
    signal = fisher_series.ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher, signal

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - trend strength indicator
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
    
    # Smooth with Wilder's method
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_sma
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    
    # Calculate 6h indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    adx = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx[i]) or np.isnan(chop[i]):
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
        htf_neutral = not htf_bull and not htf_bear
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = long signal
        # Fisher crosses below +1.5 from above = short signal
        fisher_long = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        fisher_short = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # Fisher extreme reversals
        fisher_extreme_long = fisher[i] < -2.0 and fisher[i] > fisher[i-1] if i > 0 else False
        fisher_extreme_short = fisher[i] > 2.0 and fisher[i] < fisher[i-1] if i > 0 else False
        
        # Fisher signal line cross
        fisher_cross_long = fisher[i] > fisher_signal[i] and fisher[i-1] <= fisher_signal[i-1] if i > 0 else False
        fisher_cross_short = fisher[i] < fisher_signal[i] and fisher[i-1] >= fisher_signal[i-1] if i > 0 else False
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 25.0  # Trending market
        adx_weak = adx[i] < 20.0    # Range market
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound
        chop_trend = chop[i] < 45.0   # Trending
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.2 if not np.isnan(vol_ratio[i]) else False
        
        # === REGIME DETECTION ===
        is_trend_regime = adx_strong and chop_trend
        is_range_regime = adx_weak and chop_range
        is_transition = not is_trend_regime and not is_range_regime
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Fisher reversals with HTF confirmation
        if is_trend_regime:
            if htf_bull and (fisher_long or fisher_cross_long):
                desired_signal = SIZE_STRONG if vol_confirmed else SIZE_BASE
            elif htf_bear and (fisher_short or fisher_cross_short):
                desired_signal = -SIZE_STRONG if vol_confirmed else -SIZE_BASE
            # Pullback entries in trend
            elif htf_bull and fisher_extreme_long:
                desired_signal = SIZE_BASE
            elif htf_bear and fisher_extreme_short:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Fisher extreme reversals only
        elif is_range_regime:
            if fisher_extreme_long and close[i] > hma_1d_aligned[i]:
                desired_signal = SIZE_BASE
            elif fisher_extreme_short and close[i] < hma_1d_aligned[i]:
                desired_signal = -SIZE_BASE
            # Mean reversion at Fisher extremes
            elif fisher[i] < -1.8 and fisher[i] > fisher[i-1] if i > 0 else False:
                desired_signal = SIZE_BASE * 0.8
            elif fisher[i] > 1.8 and fisher[i] < fisher[i-1] if i > 0 else False:
                desired_signal = -SIZE_BASE * 0.8
        
        # TRANSITION REGIME: Require stronger signals
        elif is_transition:
            if htf_bull and fisher_extreme_long and vol_confirmed:
                desired_signal = SIZE_BASE
            elif htf_bear and fisher_extreme_short and vol_confirmed:
                desired_signal = -SIZE_BASE
            # Fisher cross with volume
            elif htf_bull and fisher_cross_long and vol_confirmed:
                desired_signal = SIZE_BASE * 0.8
            elif htf_bear and fisher_cross_short and vol_confirmed:
                desired_signal = -SIZE_BASE * 0.8
        
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