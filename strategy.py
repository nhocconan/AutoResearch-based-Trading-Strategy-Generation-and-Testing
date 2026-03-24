#!/usr/bin/env python3
"""
Experiment #206: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime

Hypothesis: Daily timeframe with weekly trend filter can capture major moves while
avoiding whipsaws. Fisher Transform excels at catching reversals in bear markets
(2022 crash, 2025 bear). Choppiness Index switches between mean-reversion (chop)
and trend-following (trend) regimes.

Key improvements over failed #194:
- LOOSEN entry conditions (that had 0 trades)
- Fisher Transform for reversals (proven in bear markets)
- Simpler regime logic (CHOP > 55 = mean revert, CHOP < 45 = trend)
- Weekly HMA(34) for major trend bias only (not hard filter)
- Position size: 0.25 base, 0.30 strong signals
- Stoploss: 2.5x ATR trailing

Target: Sharpe > 0.399 (beat current best), trades >= 20 train, trades >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_regime_1w_v1"
timeframe = "1d"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals at extreme values (-2 to +2)
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(close) if 'close' in dir() else len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Use (high + low) / 2 as price input
    hl2 = (high + low) / 2.0
    
    # Normalize to range -1 to +1
    fisher_raw = np.zeros(n)
    fisher_raw[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(hl2[i-period+1:i+1])
        lowest = np.min(hl2[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val > 1e-10:
            normalized = 2.0 * (hl2[i] - lowest) / range_val - 1.0
            # Clamp to avoid division issues
            normalized = max(-0.999, min(0.999, normalized))
            fisher_raw[i] = normalized
    
    # Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period + 1, n):
        if not np.isnan(fisher_raw[i]) and not np.isnan(fisher_raw[i-1]):
            fisher[i] = 0.5 * np.log((1.0 + fisher_raw[i]) / (1.0 - fisher_raw[i]))
            fisher_prev[i] = 0.5 * np.log((1.0 + fisher_raw[i-1]) / (1.0 - fisher_raw[i-1]))
    
    return fisher, fisher_prev

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
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=34)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    rsi = calculate_rsi(close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after indicators are ready
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
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range-bound market → mean revert
        is_trending = chop[i] < 45.0  # Trending market → trend follow
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = False
        fisher_short = False
        
        if not np.isnan(fisher[i]) and not np.isnan(fisher_prev[i]):
            # Long: Fisher crosses above -1.5 from below (oversold reversal)
            if fisher_prev[i] < -1.5 and fisher[i] >= -1.5:
                fisher_long = True
            # Short: Fisher crosses below +1.5 from above (overbought reversal)
            if fisher_prev[i] > 1.5 and fisher[i] <= 1.5:
                fisher_short = True
        
        # === RSI EXTREMES (for choppy regime) ===
        rsi_oversold = not np.isnan(rsi[i]) and rsi[i] < 35.0
        rsi_overbought = not np.isnan(rsi[i]) and rsi[i] > 65.0
        
        # === HMA TREND ===
        hma_bull = close[i] > hma_21[i] and hma_21[i] > hma_50[i]
        hma_bear = close[i] < hma_21[i] and hma_21[i] < hma_50[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with Fisher + RSI)
        if is_choppy:
            # Long: Fisher reversal + RSI oversold + above SMA200
            if fisher_long and rsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            
            # Short: Fisher reversal + RSI overbought + below SMA200
            elif fisher_short and rsi_overbought and below_sma200:
                desired_signal = -SIZE_BASE
        
        # REGIME 2: TRENDING (trend follow with HMA + HTF)
        elif is_trending:
            # Long: HMA bull + HTF bull + Fisher not overbought
            if hma_bull and htf_bull and not fisher_short:
                # Enter on Fisher long or pullback
                if fisher_long or (rsi_oversold and above_sma200):
                    desired_signal = SIZE_STRONG
            
            # Short: HMA bear + HTF bear + Fisher not oversold
            elif hma_bear and htf_bear and not fisher_long:
                # Enter on Fisher short or bounce
                if fisher_short or (rsi_overbought and below_sma200):
                    desired_signal = -SIZE_STRONG
        
        # REGIME 3: TRANSITION (45-55 chop) - use HTF bias only
        else:
            # Simple HTF-following with Fisher confirmation
            if htf_bull and fisher_long:
                desired_signal = SIZE_BASE * 0.8
            elif htf_bear and fisher_short:
                desired_signal = -SIZE_BASE * 0.8
        
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
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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
                # Flip position
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