#!/usr/bin/env python3
"""
Experiment #532: 12h Primary + 1d HTF — Ehlers Fisher Transform + HMA Trend + Regime Adaptive

Hypothesis: 12h timeframe with Ehlers Fisher Transform provides superior reversal detection
vs RSI/CRSI during bear/range markets (2022 crash, 2025 test period). Fisher normalizes
price into Gaussian distribution, making extremes at +/-1.5 statistically significant.
Combined with HMA trend filter and 1d HTF bias, this should work in both trending and
ranging regimes without the whipsaw of pure trend-following.

Key differences from failed #524 (12h_donchian_crsi):
1. Fisher Transform instead of CRSI - better reversal detection at extremes
2. HMA(50) trend filter instead of Donchian breakout - fewer false signals
3. Regime-adaptive: trend-follow when ADX>25, mean-revert when ADX<20
4. Simpler entry logic = more trades (avoid 0-trade failure of #526)
5. 1d HMA(21) for macro bias alignment

Strategy logic:
1. 1d HMA(21) = macro trend bias (align with higher timeframe)
2. 12h HMA(50) = medium trend filter
3. 12h Fisher(9) = reversal signal (cross above -1.5 = long, cross below +1.5 = short)
4. 12h ADX(14) = regime detection (ADX>25 = trend, ADX<20 = range)
5. 12h ATR(14)*2.5 = stoploss on all positions

Regime-adaptive entries:
- TREND (ADX>25): Fisher reversal WITH trend direction only
- RANGE (ADX<20): Fisher reversal at extremes (fade the move)
- Size: 0.25 base, 0.30 strong confirmation

Target: Sharpe>0.40, trades>=30 train (7.5/year), trades>=3 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price into Gaussian normal distribution
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    
    Reference: Ehlers, J.F. "Rocket Science for Traders"
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Find highest high and lowest low over lookback
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (close[i] - lowest) / price_range - 1.0
        
        # Clamp to avoid division by zero in fisher calculation
        normalized = max(-0.999, min(0.999, normalized))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value (Ehlers uses 0.67 weighting)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
        
        fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, fisher_prev

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA with less lag"""
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

def calculate_dmi(high, low, close, period=14):
    """
    Directional Movement Index (DMI) - calculates +DI, -DI, and ADX
    ADX > 25 = trending market, ADX < 20 = ranging market
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
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
    
    return plus_di, minus_di, adx

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    hma_50 = calculate_hma(close, period=50)
    plus_di, minus_di, adx = calculate_dmi(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_50[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d macro) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === HMA TREND FILTER (12h) ===
        hma_bull = close[i] > hma_50[i]
        hma_bear = close[i] < hma_50[i]
        
        # HMA slope
        hma_slope_bull = hma_50[i] > hma_50[i-10] if i >= 10 and not np.isnan(hma_50[i-10]) else False
        hma_slope_bear = hma_50[i] < hma_50[i-10] if i >= 10 and not np.isnan(hma_50[i-10]) else False
        
        # === ADX REGIME DETECTION ===
        adx_trend = adx[i] > 25.0   # Trending market
        adx_range = adx[i] < 20.0   # Range market
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # Fisher extreme levels (for range regime)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME (ADX > 25): Only trade Fisher reversals WITH trend
        if adx_trend:
            # Long: HTF bull + HMA bull + Fisher long cross
            if htf_bull and hma_bull and fisher_long_cross:
                desired_signal = SIZE_STRONG
            # Also enter on pullback in uptrend (Fisher oversold but trend intact)
            elif htf_bull and hma_bull and fisher_oversold and fisher[i] > fisher_prev[i]:
                desired_signal = SIZE_BASE
            
            # Short: HTF bear + HMA bear + Fisher short cross
            elif htf_bear and hma_bear and fisher_short_cross:
                desired_signal = -SIZE_STRONG
            # Also enter on rally in downtrend (Fisher overbought but trend intact)
            elif htf_bear and hma_bear and fisher_overbought and fisher[i] < fisher_prev[i]:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME (ADX < 20): Mean revert at Fisher extremes
        elif adx_range:
            # Long at oversold extreme (fade the drop)
            if fisher_long_cross or (fisher_oversold and fisher[i] > fisher_prev[i]):
                desired_signal = SIZE_BASE
            # Short at overbought extreme (fade the rally)
            elif fisher_short_cross or (fisher_overbought and fisher[i] < fisher_prev[i]):
                desired_signal = -SIZE_BASE
        
        # TRANSITION REGIME (20 <= ADX <= 25): Reduced size, need HTF confirmation
        else:
            if htf_bull and fisher_long_cross:
                desired_signal = SIZE_BASE * 0.8
            elif htf_bear and fisher_short_cross:
                desired_signal = -SIZE_BASE * 0.8
            elif htf_bull and fisher_oversold and fisher[i] > fisher_prev[i]:
                desired_signal = SIZE_BASE * 0.6
            elif htf_bear and fisher_overbought and fisher[i] < fisher_prev[i]:
                desired_signal = -SIZE_BASE * 0.6
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trailing stop
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trailing stop
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