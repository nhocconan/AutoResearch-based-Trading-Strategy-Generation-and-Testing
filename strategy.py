#!/usr/bin/env python3
"""
Experiment #443: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime

Hypothesis: 6h timeframe needs better reversal detection than RSI for bear/range markets.
Fisher Transform (Ehlers) excels at catching turning points with less lag than RSI.
Choppiness Index is superior to ADX for distinguishing trend vs range regimes.

Key innovations vs #435:
1. FISHER TRANSFORM instead of RSI - catches reversals 1-2 bars earlier
2. CHOPPINESS INDEX instead of ADX+BB - single metric, more reliable regime filter
3. 1d/1w HTF instead of 12h/1d - cleaner weekly trend bias
4. Asymmetric entries - long only in bull regime, short only in bear regime
5. Streak-based exit - reduce position after 3 consecutive bars in profit

Entry Logic:
- Bull Regime (1d HMA up + 1w HMA up): Fisher < -1.5 + CHOP > 61.8 → long
- Bear Regime (1d HMA down + 1w HMA down): Fisher > +1.5 + CHOP > 61.8 → short
- Trend Breakout: CHOP < 38.2 + price breaks Donchian(20) → follow trend

Target: Sharpe>0.45, DD>-35%, trades>=60 train (15/year), trades>=10 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals when Fisher crosses extreme levels (-1.5, +1.5)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Normalize to 0-1 range
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            continue
        
        normalized = (hl2 - lowest_low) / range_val
        
        # Clamp to avoid log(0) or log(inf)
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher calculation
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            continue
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
        
        # CHOP formula
        chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # 6h HMA for trend confirmation
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss and profit taking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    profit_bars = 0  # consecutive bars in profit
    highest_profit = 0.0  # track highest profit for trailing
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
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
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = choppy/range (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        # 38.2-61.8 = transition (no new entries)
        
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === HTF TREND BIAS (1d + 1w must agree) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Both HTF must agree for strong bias
        htf_both_bull = htf_1d_bull and htf_1w_bull
        htf_both_bear = htf_1d_bear and htf_1w_bear
        
        # === PRIMARY TREND (6h HMA + SMA50/200) ===
        primary_bull = close[i] > hma_6h[i] and close[i] > sma_50[i]
        primary_bear = close[i] < hma_6h[i] and close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher < -1.5 = oversold (long signal)
        # Fisher > +1.5 = overbought (short signal)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher crossover (more responsive)
        fisher_cross_long = False
        fisher_cross_short = False
        if not np.isnan(fisher_prev[i]) and not np.isnan(fisher[i-1]):
            if fisher[i-1] <= -1.5 and fisher[i] > -1.5:
                fisher_cross_long = True
            if fisher[i-1] >= 1.5 and fisher[i] < 1.5:
                fisher_cross_short = True
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with Fisher)
        if is_choppy:
            # Long: HTF bull bias + Fisher oversold + above SMA200
            if htf_both_bull and fisher_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: HTF bear bias + Fisher overbought + below SMA200
            elif htf_both_bear and fisher_overbought and below_sma200:
                desired_signal = -SIZE_BASE
            
            # Fisher crossover entries (more aggressive)
            elif htf_both_bull and fisher_cross_long and above_sma200:
                desired_signal = SIZE_BASE
            elif htf_both_bear and fisher_cross_short and below_sma200:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (Donchian breakout with HTF confirmation)
        elif is_trending:
            # Long: HTF bull + primary bull + Donchian breakout
            if htf_both_bull and primary_bull and donchian_breakout_long:
                desired_signal = SIZE_STRONG
            
            # Short: HTF bear + primary bear + Donchian breakdown
            elif htf_both_bear and primary_bear and donchian_breakdown_short:
                desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === PROFIT TAKING (reduce after 3 consecutive profit bars) ===
        if in_position and desired_signal != 0.0:
            if position_side > 0:
                current_profit = (close[i] - entry_price) / entry_atr
                if current_profit > highest_profit:
                    highest_profit = current_profit
                    profit_bars = 0
                elif current_profit > 1.5:  # 1.5R profit
                    profit_bars += 1
                    if profit_bars >= 3:
                        desired_signal = desired_signal * 0.5  # reduce to half
            elif position_side < 0:
                current_profit = (entry_price - close[i]) / entry_atr
                if current_profit > highest_profit:
                    highest_profit = current_profit
                    profit_bars = 0
                elif current_profit > 1.5:  # 1.5R profit
                    profit_bars += 1
                    if profit_bars >= 3:
                        desired_signal = desired_signal * 0.5  # reduce to half
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) > 0.0:
            final_signal = desired_signal
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                profit_bars = 0
                highest_profit = 0.0
                # Set stoploss
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
                profit_bars = 0
                highest_profit = 0.0
        
        signals[i] = final_signal
    
    return signals