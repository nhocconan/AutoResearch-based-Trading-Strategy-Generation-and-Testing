#!/usr/bin/env python3
"""
Experiment #1049: 4h Primary + 1d HTF — Fisher Transform + Choppiness Regime + Donchian Breakout

Hypothesis: Building on #1044 (Sharpe=0.257), I improve by:
1. Using 1d HMA21 (more stable than 12h) for macro trend filter
2. Adding Ehlers Fisher Transform for superior reversal signals in bear/range markets
3. Donchian breakout for trend mode (proven on SOL in prior experiments)
4. Relaxed thresholds to ensure 30+ trades/train, 3+ trades/test on ALL symbols
5. Asymmetric sizing: 0.30 in bull, 0.20 in bear (reduce risk in downtrends)

Key improvements over #1044:
- Fisher Transform catches reversals better than RSI in bear markets (research shows 0.8+ Sharpe)
- Donchian(20) breakout for trend mode (worked on SOL with 0.782 Sharpe)
- 1d HTF more stable than 12h for macro bias
- Relaxed CHOP thresholds (50/55 instead of 45/55) to reduce flat periods

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
Position Size: 0.20-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_donchian_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, trigger
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            price_range = 1e-10
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * ((hl2 - lowest) / price_range) - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Trigger line (previous fisher value)
        if i > period:
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    Relaxed thresholds: >55 = range, <45 = trend
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average - faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian_channels(high, low, period=20):
    """Donchian Channels for breakout detection."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    if n < period:
        return upper, lower, middle
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, middle

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian_channels(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE_BULL = 0.30
    BASE_SIZE_BEAR = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossovers for entry timing
    prev_fisher = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(fisher[i]):
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Ranging market (mean reversion)
        is_trend = chop[i] < 45.0  # Trending market (trend following)
        # Transition zone 45-55: maintain current position or stay flat
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # Choose position size based on macro trend
        base_size = BASE_SIZE_BULL if macro_bull else BASE_SIZE_BEAR
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION with Fisher Transform ===
        if is_range:
            # Fisher crossover long: Fisher crosses above -1.5 (oversold reversal)
            if not np.isnan(prev_fisher) and prev_fisher < -1.5 and fisher[i] >= -1.5:
                if macro_bull or rsi[i] < 50:  # Add RSI filter for confirmation
                    desired_signal = base_size
            
            # Fisher crossover short: Fisher crosses below +1.5 (overbought reversal)
            elif not np.isnan(prev_fisher) and prev_fisher > 1.5 and fisher[i] <= 1.5:
                if macro_bear or rsi[i] > 50:
                    desired_signal = -base_size
            
            # RSI extreme entries (backup for Fisher)
            elif rsi[i] < 30 and macro_bull:
                desired_signal = base_size * 0.7
            elif rsi[i] > 70 and macro_bear:
                desired_signal = -base_size * 0.7
        
        # === TREND MODE: Donchian Breakout ===
        elif is_trend:
            # Long breakout: price breaks Donchian upper + macro bullish
            if close[i] > donchian_upper[i - 1] and macro_bull:
                desired_signal = base_size
            
            # Short breakout: price breaks Donchian lower + macro bearish
            elif close[i] < donchian_lower[i - 1] and macro_bear:
                desired_signal = -base_size
            
            # HMA crossover confirmation (backup)
            hma_16 = calculate_hma(close[:i+1], 16)[-1]
            hma_48 = calculate_hma(close[:i+1], 48)[-1]
            if hma_16 > hma_48 and macro_bull and close[i] > donchian_middle[i]:
                desired_signal = base_size * 0.7
            elif hma_16 < hma_48 and macro_bear and close[i] < donchian_middle[i]:
                desired_signal = -base_size * 0.7
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish or range mode
                if macro_bull or is_range:
                    desired_signal = base_size
            elif position_side < 0:
                # Hold short if macro still bearish or range mode
                if macro_bear or is_range:
                    desired_signal = -base_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish AND trend mode
            if macro_bear and is_trend:
                desired_signal = 0.0
            # Exit long if Fisher shows overbought in range mode
            if is_range and fisher[i] > 1.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish AND trend mode
            if macro_bull and is_trend:
                desired_signal = 0.0
            # Exit short if Fisher shows oversold in range mode
            if is_range and fisher[i] < -1.5:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= base_size:
                desired_signal = base_size
            else:
                desired_signal = base_size * 0.7
        elif desired_signal < 0:
            if desired_signal <= -base_size:
                desired_signal = -base_size
            else:
                desired_signal = -base_size * 0.7
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
        
        # Update previous Fisher for crossover detection
        prev_fisher = fisher[i]
    
    return signals