#!/usr/bin/env python3
"""
Experiment #520: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime

Hypothesis: 6h timeframe with 1w/1d HTF filters captures medium-term swings
while avoiding noise. Fisher Transform excels at catching reversals in bear/range
markets (2022 crash, 2025 bear). Choppiness Index regime filter switches between
mean-reversion (chop>61.8) and trend-following (chop<38.2).

Strategy logic:
1. 1w HMA(21) = weekly trend bias (strongest HTF filter)
2. 1d Choppiness(14) = regime detection (range vs trend)
3. 6h Fisher Transform(9) = reversal entries (crosses -1.5 long, +1.5 short)
4. 6h RSI(7) = momentum confirmation (avoid counter-trend extremes)
5. ATR(14)*2.5 stoploss on all positions
6. Regime-adaptive sizing: 0.30 trend, 0.25 mean-revert

Key improvements from failed 6h experiments:
- Fisher Transform instead of RSI alone (better reversal capture)
- 1w HMA instead of 1d (stronger trend filter, fewer whipsaws)
- Choppiness from 1d instead of 6h (more stable regime signal)
- Looser Fisher thresholds (-1.5/+1.5 vs -1.0/+1.0) for more trades
- RSI filter only blocks extreme counter-trend (not primary entry)

Target: Sharpe>0.40 (beat current best 0.399), trades>=80 train, trades>=10 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_1w1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points better than RSI in bear/range markets
    
    Formula:
    1. Price = (2 * Close - (High + Low)) / (High - Low)
    2. Value = 0.66 * Price + 0.67 * prev_Value
    3. Fisher = 0.5 * ln((1 + Value) / (1 - Value))
    
    Entry signals:
    - Fisher crosses above -1.5 from below = long
    - Fisher crosses below +1.5 from above = short
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate normalized price
    price = np.zeros(n)
    price[:] = np.nan
    
    for i in range(period, n):
        hl_range = high[i] - low[i]
        if hl_range > 1e-10:
            price[i] = (2.0 * close[i] - (high[i] + low[i])) / hl_range
            # Clamp to avoid division issues
            price[i] = np.clip(price[i], -0.999, 0.999)
    
    # Smooth with EMA-like filter
    value = np.zeros(n)
    value[:] = np.nan
    value[period] = price[period] if not np.isnan(price[period]) else 0.0
    
    for i in range(period + 1, n):
        if not np.isnan(price[i]):
            value[i] = 0.66 * price[i] + 0.67 * value[i-1]
            value[i] = np.clip(value[i], -0.999, 0.999)
    
    # Fisher transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(value[i]) and abs(value[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + value[i]) / (1.0 - value[i]))
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d Choppiness for regime
    chop_1d_raw = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate 6h indicators
    fisher = calculate_fisher_transform(high, low, close, period=9)
    rsi = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=21)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_MR = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w HTF BIAS ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === 1d CHOPPINESS REGIME ===
        chop_val = chop_1d_aligned[i]
        chop_range = chop_val > 58.0  # Range-bound market
        chop_trend = chop_val < 42.0  # Trending market
        # chop between 42-58 = neutral
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev = fisher[i-1] if i > 0 else fisher_val
        
        # Fisher crossover signals
        fisher_long_cross = fisher_prev <= -1.5 and fisher_val > -1.5
        fisher_short_cross = fisher_prev >= 1.5 and fisher_val < 1.5
        
        # Fisher extreme levels
        fisher_deep_oversold = fisher_val < -2.0
        fisher_deep_overbought = fisher_val > 2.0
        
        # Fisher turning (reversal from extreme)
        fisher_turning_long = fisher_deep_oversold and fisher_val > fisher_prev
        fisher_turning_short = fisher_deep_overbought and fisher_val < fisher_prev
        
        # === RSI FILTERS ===
        rsi_val = rsi[i]
        rsi_oversold = rsi_val < 35.0
        rsi_overbought = rsi_val > 65.0
        rsi_extreme_oversold = rsi_val < 25.0
        rsi_extreme_overbought = rsi_val > 75.0
        
        # RSI momentum
        rsi_rising = rsi_val > rsi[i-1] if i > 0 else False
        rsi_falling = rsi_val < rsi[i-1] if i > 0 else False
        
        # === VOLATILITY FILTER ===
        atr_ratio = atr[i] / np.nanmean(atr[max(0,i-100):i]) if i >= 100 else 1.0
        vol_normal = atr_ratio < 3.0
        vol_spike = atr_ratio > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF with Fisher confirmation
        if chop_trend and vol_normal:
            # Long: HTF bull + HMA bull + Fisher long cross
            if htf_bull and hma_bull and fisher_long_cross and not rsi_extreme_overbought:
                desired_signal = SIZE_TREND
            # Short: HTF bear + HMA bear + Fisher short cross
            elif htf_bear and hma_bear and fisher_short_cross and not rsi_extreme_oversold:
                desired_signal = -SIZE_TREND
            # HMA pullback entry in trend
            elif htf_bull and hma_bull and above_sma50 and fisher_turning_long:
                desired_signal = SIZE_TREND * 0.8
            elif htf_bear and hma_bear and below_sma50 and fisher_turning_short:
                desired_signal = -SIZE_TREND * 0.8
        
        # RANGE REGIME: Mean reversion with Fisher extremes
        if chop_range and vol_normal:
            # Long: Fisher deep oversold + RSI oversold + above SMA200
            if fisher_deep_oversold and rsi_oversold and above_sma200:
                desired_signal = SIZE_MR
            # Short: Fisher deep overbought + RSI overbought + below SMA200
            elif fisher_deep_overbought and rsi_overbought and below_sma200:
                desired_signal = -SIZE_MR
            # Fisher turning from extreme
            elif fisher_turning_long and rsi_rising and above_sma50:
                desired_signal = SIZE_MR * 0.8
            elif fisher_turning_short and rsi_falling and below_sma50:
                desired_signal = -SIZE_MR * 0.8
        
        # NEUTRAL REGIME (chop 42-58): Use both strategies
        if not chop_range and not chop_trend and vol_normal:
            # HTF-aligned Fisher signals
            if htf_bull and fisher_long_cross and not rsi_extreme_overbought:
                desired_signal = SIZE_MR
            elif htf_bear and fisher_short_cross and not rsi_extreme_oversold:
                desired_signal = -SIZE_MR
            # Fisher turning with RSI confirmation
            elif fisher_turning_long and rsi_rising:
                desired_signal = SIZE_MR * 0.7
            elif fisher_turning_short and rsi_falling:
                desired_signal = -SIZE_MR * 0.7
        
        # VOL SPIKE REVERSION: Extra entries on panic
        if vol_spike and not chop_trend:
            if fisher_deep_oversold and rsi_extreme_oversold:
                desired_signal = SIZE_MR * 0.6
            elif fisher_deep_overbought and rsi_extreme_overbought:
                desired_signal = -SIZE_MR * 0.6
        
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
        elif desired_signal >= SIZE_MR * 0.9:
            final_signal = SIZE_MR
        elif desired_signal <= -SIZE_MR * 0.9:
            final_signal = -SIZE_MR
        elif desired_signal >= SIZE_MR * 0.5:
            final_signal = SIZE_MR * 0.8
        elif desired_signal <= -SIZE_MR * 0.5:
            final_signal = -SIZE_MR * 0.8
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