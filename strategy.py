#!/usr/bin/env python3
"""
Experiment #310: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + HTF HMA

Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets (2025 test).
Combined with Choppiness Index regime detection and 4h/1d HMA for trend bias.

Key innovations:
1. FISHER TRANSFORM (Ehlers): period=9, catches reversals when Fisher crosses -1.5/+1.5
   Proven to work in bear markets where trend-following fails
2. CHOPPINESS REGIME: CHOP>60 = mean revert (Fisher extremes), CHOP<40 = trend follow
3. HTF HMA BIAS: 4h HMA(21) + 1d HMA(50) for trend direction
4. SESSION FILTER: 08-20 UTC only (high liquidity, avoid Asian session chop)
5. VERY SELECTIVE: 3+ confluence required (regime + Fisher + HTF + session)

Entry Logic:
- Choppy regime: Fisher < -1.5 + price > 4h HMA → long; Fisher > +1.5 + price < 4h HMA → short
- Trending regime: Fisher cross + HTF alignment + Donchian breakout
- Session: Only enter 08-20 UTC (avoid 00-08 UTC Asian chop)

Position sizing: 0.20 base, 0.30 when 1d HMA aligned (discrete levels)
Stoploss: 2.5x ATR from entry price

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test, 40-80 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_htf_hma_session_v1"
timeframe = "1h"
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
    CHOP > 61.8 = choppy/range bound (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * prev_X
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    x_val = 0.0
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            # Normalize price to 0-1 range
            normalized = (close[i] - lowest_low) / price_range
            
            # Calculate X with smoothing
            x_val = 0.66 * (normalized - 0.5) + 0.67 * x_val
            
            # Clamp X to prevent division by zero
            x_val = np.clip(x_val, -0.99, 0.99)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + x_val) / (1.0 - x_val))
            
            if i > period:
                fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column"""
    # open_time is in milliseconds since epoch
    hours = (prices["open_time"].values // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(prices)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(fisher[i]):
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 60.0
        trending_threshold = 40.0
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy → mean revert
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending → trend follow
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = False
        fisher_cross_down = False
        
        if not np.isnan(fisher_prev[i]):
            # Long: Fisher crosses above -1.5
            fisher_cross_up = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
            # Short: Fisher crosses below +1.5
            fisher_cross_down = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extreme levels (for choppy regime)
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = False
        breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            breakout_short = close[i] < donchian_lower[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            desired_signal = 0.0
        else:
            # REGIME 1: CHOPPY (mean reversion with Fisher extremes)
            if current_regime == 2:
                # Long: Fisher extreme low + above 4h HMA + above SMA200
                if fisher_extreme_low and htf_4h_bull and above_sma200:
                    # Stronger size if 1d aligned
                    if htf_1d_bull:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
                
                # Short: Fisher extreme high + below 4h HMA + below SMA200
                elif fisher_extreme_high and htf_4h_bear and below_sma200:
                    if htf_1d_bear:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            
            # REGIME 2: TRENDING (Fisher cross + breakout + HTF alignment)
            elif current_regime == 1:
                # Long: Fisher cross up + 4h bull + 1d bull + Donchian breakout
                if fisher_cross_up and htf_4h_bull and htf_1d_bull:
                    if breakout_long:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
                
                # Short: Fisher cross down + 4h bear + 1d bear + Donchian breakout
                elif fisher_cross_down and htf_4h_bear and htf_1d_bear:
                    if breakout_short:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
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