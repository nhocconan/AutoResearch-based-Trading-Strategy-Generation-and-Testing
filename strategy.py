#!/usr/bin/env python3
"""
Experiment #250: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + Session Filter

Hypothesis: 1h timeframe with Fisher Transform for reversals + Choppiness regime detection
can capture both mean-reversion in choppy markets AND trend pullbacks in trending markets.

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, catches reversals faster than RSI in bear markets
   Long when Fisher crosses above -1.5 from below, Short when crosses below +1.5 from above
2. CHOPPINESS INDEX regime filter: CHOP>60 = mean revert, CHOP<40 = trend follow
3. 4h HMA(21) for trend bias - only trade with HTF direction
4. SESSION FILTER: 08-20 UTC only (high liquidity, avoid Asia overnight whipsaw)
5. 1d HMA(50) as secondary filter for stronger trend confirmation

Why this should work:
- Fisher Transform excels at identifying turning points in ranging/bear markets (BTC 2022, 2025)
- Choppiness prevents trend strategies from whipsawing in ranges
- Session filter reduces false breakouts during low liquidity
- 4h/1d HTF ensures we trade with major trend, not against it
- Conservative size (0.20) + ATR stops control drawdown

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test, trades/year=40-80
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_4h1d_session_v1"
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
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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
    Highlights turning points better than RSI in ranging markets
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) * 2 - 1
    3. Apply Fisher: 0.5 * ln((1 + x) / (1 - x))
    4. Smooth with EMA
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize price to -1 to +1 range
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized[i] = ((typical[i] - lowest) / price_range) * 2.0 - 1.0
            # Clamp to avoid division by zero in Fisher
            normalized[i] = np.clip(normalized[i], -0.999, 0.999)
    
    # Fisher Transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(normalized[i]):
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
    
    # Smooth Fisher with EMA
    fisher_smooth = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    # Fisher trigger (previous bar value for signal generation)
    fisher_trigger = np.roll(fisher_smooth, 1)
    fisher_trigger[:period+1] = np.nan
    
    return fisher_smooth, fisher_trigger

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # Convert ms to seconds, then to datetime
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Get session hours
    session_hours = get_session_hour(open_time)
    
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
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
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
        
        # === SESSION FILTER: Only trade 08-20 UTC ===
        current_hour = session_hours[i]
        in_session = (current_hour >= 8) and (current_hour <= 20)
        
        if not in_session:
            # Close existing positions outside session
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 60.0
        trending_threshold = 40.0
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy - mean revert
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending - trend follow
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF alignment (both 4h and 1d agree)
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_signal = False
        fisher_short_signal = False
        
        # Long: Fisher crosses above -1.5 from below
        if not np.isnan(fisher_trigger[i]) and not np.isnan(fisher[i]):
            if fisher_trigger[i] < -1.5 and fisher[i] >= -1.5:
                fisher_long_signal = True
            # Short: Fisher crosses below +1.5 from above
            elif fisher_trigger[i] > 1.5 and fisher[i] <= 1.5:
                fisher_short_signal = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with Fisher)
        if current_regime == 2:
            # Long: Fisher reversal + above SMA200 + HTF not strongly bearish
            if fisher_long_signal and above_sma200 and not htf_strong_bear:
                desired_signal = SIZE_BASE
            
            # Short: Fisher reversal + below SMA200 + HTF not strongly bullish
            elif fisher_short_signal and below_sma200 and not htf_strong_bull:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (pullback entries with Fisher confirmation)
        elif current_regime == 1:
            # Long: HTF bullish + Fisher confirms pullback entry
            if htf_strong_bull and fisher_long_signal:
                desired_signal = SIZE_STRONG
            elif htf_4h_bull and fisher_long_signal and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: HTF bearish + Fisher confirms pullback entry
            elif htf_strong_bear and fisher_short_signal:
                desired_signal = -SIZE_STRONG
            elif htf_4h_bear and fisher_short_signal and below_sma200:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
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