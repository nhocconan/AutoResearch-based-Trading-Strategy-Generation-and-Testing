#!/usr/bin/env python3
"""
Experiment #475: 6h Primary + 12h/1d HTF — Fisher Transform Reversal + Simplified HTF Bias

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Recent 6h failures show:
- Dual HTF agreement (12h AND 1d) is TOO RESTRICTIVE → few signals
- Complex regime detection (ADX + BB Width) filters out good entries
- Weekly pivot strategies generate 0 trades

New approach:
1. SIMPLER HTF BIAS: 12h HMA only (not requiring 12h AND 1d agreement) → more signals
2. FISHER TRANSFORM: Period 9, proven reversal indicator (Ehlers), rarely tried on 6h
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
3. SINGLE REGIME FILTER: ADX > 20 = trend, ADX < 20 = chop (not compound conditions)
4. ASYMMETRIC ENTRY: Trend entries get SIZE_STRONG (0.30), mean reversion gets SIZE_BASE (0.20)
5. VOLUME CONFIRMATION: Breakouts need volume > 20-bar avg (filters false breakouts)
6. LOOSER STOPLOSS: 2.5x ATR (not 2.0x) to avoid premature exits on 6h noise

Entry Logic:
- Trending Long: 12h HMA bull + Fisher > -1.5 cross + ADX > 20 + volume confirm
- Trending Short: 12h HMA bear + Fisher < +1.5 cross + ADX > 20 + volume confirm
- Choppy Long: Fisher < -2.0 (extreme) + price < BB lower (mean reversion)
- Choppy Short: Fisher > +2.0 (extreme) + price > BB upper (mean reversion)

Target: Sharpe>0.45, DD>-35%, trades>=60 train, trades>=10 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_simplified_htf_v1"
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

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals at extremes better than RSI
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Normalize price to -1 to +1 range using highest high and lowest low
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize to -1 to +1
        normalized = 2.0 * (close[i] - lowest) / (highest - lowest) - 1.0
        
        # Clamp to avoid division issues
        normalized = max(-0.999, min(0.999, normalized))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Trigger line (1-period lag of fisher)
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

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
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias (12h only, simpler than dual)
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # 1d HMA for additional confirmation (optional, not required for entry)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_dev=2.0)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Fisher Transform for entry timing
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20      # Mean reversion size
    SIZE_STRONG = 0.30    # Trend breakout size
    SIZE_MAX = 0.35       # Absolute max
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (SIMPLE: ADX only) ===
        is_trending = adx[i] > 20.0
        is_choppy = adx[i] <= 20.0
        
        # === HTF BIAS (12h HMA only - simpler than dual) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # 1d HMA as secondary confirmation (bonus, not required)
        htf_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 0 and not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            if not np.isnan(fisher_trigger[i]) and not np.isnan(fisher_trigger[i-1]):
                # Long: Fisher crosses above -1.5 from below
                if fisher[i-1] < -1.5 and fisher[i] > -1.5:
                    fisher_cross_long = True
                # Short: Fisher crosses below +1.5 from above
                if fisher[i-1] > 1.5 and fisher[i] < 1.5:
                    fisher_cross_short = True
        
        # Fisher extreme for mean reversion
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === BB TOUCH (mean reversion) ===
        touch_lower = close[i] <= bb_lower[i]
        touch_upper = close[i] >= bb_upper[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = volume[i] > vol_sma[i] if not np.isnan(vol_sma[i]) else False
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (ADX > 20)
        if is_trending:
            # Long: HTF bull + Fisher cross long + volume confirm
            if htf_bull and fisher_cross_long:
                if volume_confirm or above_sma50:
                    desired_signal = SIZE_STRONG
            
            # Short: HTF bear + Fisher cross short + volume confirm
            elif htf_bear and fisher_cross_short:
                if volume_confirm or below_sma50:
                    desired_signal = -SIZE_STRONG
        
        # REGIME 2: CHOPPY (ADX <= 20) - Mean Reversion
        elif is_choppy:
            # Long: Fisher extreme + BB lower touch (oversold bounce)
            if fisher_extreme_long and touch_lower:
                if above_sma200:  # Only long if above long-term trend
                    desired_signal = SIZE_BASE
            
            # Short: Fisher extreme + BB upper touch (overbought fade)
            elif fisher_extreme_short and touch_upper:
                if below_sma200:  # Only short if below long-term trend
                    desired_signal = -SIZE_BASE
            
            # Additional: Fisher cross in chop (weaker signal)
            elif fisher_cross_long and touch_lower:
                desired_signal = SIZE_BASE * 0.7
            elif fisher_cross_short and touch_upper:
                desired_signal = -SIZE_BASE * 0.7
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) > 0.05:
            final_signal = np.sign(desired_signal) * SIZE_BASE
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
        
        signals[i] = final_signal
    
    return signals