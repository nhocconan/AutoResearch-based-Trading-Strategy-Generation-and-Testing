#!/usr/bin/env python3
"""
Experiment #1081: 4h Primary + 1d/1w HTF — Fisher Transform + KAMA + Choppiness Regime

Hypothesis: After 780+ failed experiments, the winning pattern combines:
1. EHLERS FISHER TRANSFORM — superior reversal detection vs RSI/CRSI/STC
   Long: Fisher crosses above -1.5 from below | Short: Fisher crosses below +1.5 from above
   Proven in bear/range markets (2022 crash, 2025 sideways)
2. KAMA (Kaufman Adaptive) — adapts smoothing to market efficiency
   Better than HMA/EMA in choppy crypto markets
3. CHOPPINESS INDEX — regime filter (keep proven component)
   CHOP > 61.8 = range (mean reversion) | CHOP < 38.2 = trend (breakout)
4. 1d HMA21 macro bias — simpler than dual HTF, more trades
5. ATR trailing stop — 2.5x for risk management

Why this should beat Sharpe=0.612:
- Fisher Transform is DIFFERENT from all 780 failed strategies (no RSI/CRSI/STC)
- KAMA adapts to volatility better than fixed EMA/HMA
- Simpler entry logic = MORE trades (avoid 0-trade failure mode)
- 4h timeframe = 20-50 trades/year target
- Conservative sizing (0.30) controls drawdown

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base, 0.15 in high vol
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_chop_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range over period
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    
    Signals:
    - Fisher < -1.5 = oversold (long opportunity)
    - Fisher > +1.5 = overbought (short opportunity)
    - Crossovers at these levels give entry signals
    
    Research: Superior to RSI for catching reversals in bear markets.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize to -1 to +1
    normalized = np.full(n, np.nan)
    for i in range(period, n):
        window = typical[i - period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        highest = np.nanmax(window)
        lowest = np.nanmin(window)
        price_range = highest - lowest
        if price_range > 1e-10:
            normalized[i] = 2.0 * (typical[i] - lowest) / price_range - 1.0
        else:
            normalized[i] = 0.0
    
    # Clamp to avoid log(0)
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher_raw = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
    
    # Smooth with EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA) — adapts to market efficiency.
    
    Formula:
    1. Change = |close - close[n]|
    2. Volatility = sum(|close[i] - close[i-1]|) over period
    3. Efficiency Ratio (ER) = Change / Volatility (0 to 1)
    4. Smoothing Constant (SC) = [ER * (fast_SC - slow_SC) + slow_SC]^2
    5. KAMA = KAMA[prev] + SC * (close - KAMA[prev])
    
    Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    
    Advantage: Smooth in trends, responsive in breakouts.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < efficiency_period + slow_period:
        return kama
    
    # Fast and slow smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA with SMA
    kama[efficiency_period - 1] = np.nanmean(close[:efficiency_period])
    
    for i in range(efficiency_period, n):
        if np.isnan(kama[i-1]):
            continue
        
        # Change over efficiency period
        change = abs(close[i] - close[i - efficiency_period])
        
        # Volatility (sum of absolute price changes)
        volatility = 0.0
        for j in range(1, efficiency_period + 1):
            volatility += abs(close[i - j + 1] - close[i - j])
        
        # Efficiency Ratio
        if volatility > 1e-10:
            er = change / volatility
        else:
            er = 0.0
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    
    CHOP > 61.8 = choppy/range (mean reversion)
    CHOP < 38.2 = trending (breakout/trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # CHOP calculation
    for i in range(period, n):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        price_range = hh[i] - ll[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR Ratio for volatility regime (spike detection)."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(len(close), np.nan)
    valid_mask = (~np.isnan(atr_short)) & (~np.isnan(atr_long)) & (atr_long > 1e-10)
    ratio[valid_mask] = atr_short[valid_mask] / atr_long[valid_mask]
    
    return ratio

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher = calculate_fisher_transform(high, low, period=9)
    kama = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossovers
    prev_fisher = np.full(n, 0.0)
    for i in range(1, n):
        if not np.isnan(fisher[i-1]):
            prev_fisher[i] = fisher[i-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(kama[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donch_upper[i]):
            continue
        
        # Volatility regime
        vol_spike = atr_ratio[i] > 2.0
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        # Macro trend (1d HMA21)
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # Choppiness regime
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # Fisher levels
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher crossovers
        fisher_long_cross = prev_fisher[i] < -1.5 and fisher[i] >= -1.5
        fisher_short_cross = prev_fisher[i] > 1.5 and fisher[i] <= 1.5
        
        # Donchian breakout
        donch_breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        donch_breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # KAMA trend
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        desired_signal = 0.0
        
        # === CHOPPY REGIME: Mean Reversion ===
        if is_choppy:
            # Long: Fisher oversold + KAMA bull + macro bull
            if fisher_oversold and kama_bull and macro_bull:
                desired_signal = current_size
            # Short: Fisher overbought + KAMA bear + macro bear
            elif fisher_overbought and kama_bear and macro_bear:
                desired_signal = -current_size
        
        # === TRENDING REGIME: Breakout Following ===
        elif is_trending:
            # Long breakout + Fisher turning up + macro bull
            if donch_breakout_long and fisher[i] > -1.0 and macro_bull:
                desired_signal = current_size
            elif fisher_long_cross and macro_bull:
                desired_signal = current_size
            
            # Short breakout + Fisher turning down + macro bear
            if donch_breakout_short and fisher[i] < 1.0 and macro_bear:
                desired_signal = -current_size
            elif fisher_short_cross and macro_bear:
                desired_signal = -current_size
        
        # === TRANSITION ZONE: Combined Signals ===
        else:
            # Long: Fisher crossover + KAMA bull + macro bull
            if fisher_long_cross and kama_bull and macro_bull:
                desired_signal = current_size
            elif fisher_oversold and kama_bull:
                desired_signal = current_size * 0.5
            
            # Short: Fisher crossover + KAMA bear + macro bear
            if fisher_short_cross and kama_bear and macro_bear:
                desired_signal = -current_size
            elif fisher_overbought and kama_bear:
                desired_signal = -current_size * 0.5
        
        # === STOPLOSS (Trailing ATR 2.5x) ===
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
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if fisher_overbought and close[i] < donch_mid[i]:
                desired_signal = 0.0
            if macro_bear and fisher[i] < 0.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if fisher_oversold and close[i] > donch_mid[i]:
                desired_signal = 0.0
            if macro_bull and fisher[i] > 0.0:
                desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if fisher[i] > -1.0 or close[i] > kama[i]:
                    desired_signal = current_size
            elif position_side < 0:
                if fisher[i] < 1.0 or close[i] < kama[i]:
                    desired_signal = -current_size
        
        # === DISCRETIZE ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        
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
    
    return signals