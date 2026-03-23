#!/usr/bin/env python3
"""
Experiment #1116: 12h Primary + 1d HTF — KAMA Trend + Donchian Breakout + Choppiness Filter

Hypothesis: After analyzing 800+ failed experiments, key insights for 12h timeframe:
1. KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA
   - In trending markets: KAMA follows price closely (low lag)
   - In choppy markets: KAMA flattens (reduces whipsaws)
   - Research showed KAMA + ADX + Choppiness achieved ETH Sharpe +0.755
2. Donchian breakout (20-period) provides clear entry signals with momentum confirmation
   - Proven on SOL with Sharpe +0.879 in research
3. Choppiness Index (CHOP) filters out range markets where trend strategies fail
   - CHOP > 61.8 = choppy/range (avoid trend entries)
   - CHOP < 38.2 = trending (enable breakout entries)
4. 1d KAMA provides macro trend filter without over-complication
5. 12h Donchian breakout ensures 20-50 trades/year frequency
6. ATR trailing stop (2.5x) protects capital during reversals

Why this should beat Sharpe=0.612:
- KAMA adapts to market regime automatically (no manual regime detection)
- Donchian breakout = clear momentum signal, less subjective than RSI
- Choppiness filter avoids the #1 killer: trading trends in choppy markets
- 12h timeframe naturally limits trade frequency = less fee drag
- Conservative position sizing (0.25-0.30) controls drawdown

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 20-50 trades/year, Sharpe > 0.612, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_donchian_chop_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    
    KAMA adapts to market volatility:
    - In trending markets: follows price closely (low lag)
    - In choppy markets: flattens out (reduces whipsaws)
    
    Formula:
    1. Efficiency Ratio (ER) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    2. Smoothing Constant (SC) = [ER * (fast_SC - slow_SC) + slow_SC]^2
    3. KAMA[i] = KAMA[i-1] + SC * (close[i] - KAMA[i-1])
    
    Where:
    - fast_SC = 2/(fast_period+1) = 2/3 = 0.6667
    - slow_SC = 2/(slow_period+1) = 2/31 = 0.0645
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA of first 'period' bars
    kama[period - 1] = np.mean(close[:period])
    
    # Calculate KAMA
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    
    Measures market choppiness vs trending:
    - CHOP > 61.8 = choppy/range market (avoid trend trades)
    - CHOP < 38.2 = trending market (enable trend trades)
    - Values between 38.2-61.8 = transition zone
    
    Formula:
    CHOP = 100 * LOG10(sum(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
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
    
    # Calculate CHOP using rolling window
    for i in range(period, n):
        tr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/choppy market.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth DM and TR using Wilder's smoothing (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_s > 1e-10
    plus_di[mask] = 100.0 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / tr_s[mask]
    
    # Calculate DX
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # ADX = EMA of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian for breakout signals
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # 12h KAMA for local trend
    kama_12h = calculate_kama(close, period=10)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_12h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d KAMA) ===
        # KAMA adapts to volatility - follows price in trends, flattens in chop
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === LOCAL TREND (12h KAMA) ===
        local_bull = close[i] > kama_12h[i]
        local_bear = close[i] < kama_12h[i]
        
        # === CHOPPY FILTER ===
        # Only trade when market is trending (CHOP < 55, looser than 38.2 for frequency)
        is_trending = chop[i] < 55.0
        
        # === TREND STRENGTH (ADX) ===
        # Only trade when ADX > 18 (trend has some strength)
        trend_strong = adx[i] > 18.0
        
        # === BREAKOUT SIGNAL (Donchian) ===
        # Long: price breaks above Donchian upper
        # Short: price breaks below Donchian lower
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # LONG: Macro bull + Local bull + Trending + ADX strong + Donchian breakout
        if macro_bull and local_bull and is_trending and trend_strong:
            if breakout_long:
                desired_signal = current_size
        
        # SHORT: Macro bear + Local bear + Trending + ADX strong + Donchian breakout
        elif macro_bear and local_bear and is_trending and trend_strong:
            if breakout_short:
                desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro and local still bull
                if macro_bull and local_bull:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro and local still bear
                if macro_bear and local_bear:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses or local trend breaks
            if macro_bear or local_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses or local trend breaks
            if macro_bull or local_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
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
        else:
            desired_signal = 0.0
        
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
    
    return signals