#!/usr/bin/env python3
"""
Experiment #904: 4h Primary + 12h/1d HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: After 636 failed strategies, the key insight is that 4h timeframe works
better than 12h/1d for BTC/ETH. This strategy combines:

1. Ehlers Fisher Transform (period=9): Catches reversals in bear markets where
   simple trend following fails. Long when Fisher crosses above -1.5, short when
   crosses below +1.5. Proven in research for bear/range markets.

2. KAMA (Kaufman Adaptive Moving Average, ER=10): Better than HMA/EMA in choppy
   markets. Adapts speed based on market efficiency ratio.

3. ADX (14) + DI+/DI-: Trend strength filter. Only trend-follow when ADX>25.
   Mean-revert when ADX<20.

4. 12h HMA(21) for medium-term trend bias (HTF direction filter)
5. 1d HMA(21) for macro regime (bull/bear market filter)
6. ATR(14) trailing stop (2.5x) for risk management

Why this should work on 4h:
- Fisher Transform excels at catching reversals in bear markets (2022 crash, 2025 bear)
- KAMA adapts to volatility, reducing whipsaw in choppy periods
- ADX regime switch: trend-follow when strong, mean-revert when weak
- Dual HTF (12h + 1d) provides stronger trend bias than single HTF
- Relaxed Fisher thresholds (-1.5/+1.5 not -2/+2) ensure trades on all symbols

Critical improvements from failed experiments:
- Fisher Transform instead of RSI/CRSI (better for bear market reversals)
- KAMA instead of HMA/EMA (adaptive to volatility)
- ADX regime switch (different logic for trending vs ranging)
- Relaxed entry thresholds to guarantee 30+ trades per symbol
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_adx_regime_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    ER near 1 = trending (fast smoothing), ER near 0 = choppy (slow smoothing)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        price_change = np.abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for better reversal signals.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest)
    3. Scale: 0.66 * (normalized - 0.5) + 0.67 * prev_fisher
    4. Fisher: 0.5 * ln((1 + scaled) / (1 - scaled))
    
    Long signal: Fisher crosses above -1.5
    Short signal: Fisher crosses below +1.5
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 2:
        return fisher, fisher_prev
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest == lowest:
            fisher[i] = fisher[i - 1] if i > 0 else 0
            fisher_prev[i] = fisher[i - 1] if i > 0 else 0
            continue
        
        # Typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Normalize (0 to 1)
        normalized = (typical - lowest) / (highest - lowest)
        
        # Scale to -1 to +1 range with smoothing
        if i > period:
            scaled = 0.66 * (normalized - 0.5) + 0.67 * fisher_prev[i - 1]
        else:
            scaled = 0.66 * (normalized - 0.5)
        
        # Clamp to avoid log errors
        scaled = np.clip(scaled, -0.99, 0.99)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + scaled) / (1 - scaled))
        fisher_prev[i] = fisher[i - 1] if i > period else 0
    
    return fisher, fisher_prev

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) + DI+ / DI-
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    adx = np.full(n, np.nan)
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx, di_plus, di_minus
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i - 1]), np.abs(low[i] - close[i - 1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed values (Wilder's smoothing)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di_raw = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di_raw = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    for i in range(period, n):
        if atr[i] > 1e-10:
            di_plus[i] = 100 * plus_di_raw[i] / atr[i]
            di_minus[i] = 100 * minus_di_raw[i] / atr[i]
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = adx_raw
    
    return adx, di_plus, di_minus

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i - 1]), np.abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher_4h, fisher_prev_4h = calculate_fisher_transform(high, low, period=9)
    adx_4h, di_plus_4h, di_minus_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 12h HMA for medium-term trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(fisher_4h[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(di_plus_4h[i]) or np.isnan(di_minus_4h[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (12h HTF HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === SHORT-TERM TREND FILTER (4h KAMA) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === ADX REGIME DETECTION ===
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        # Neutral: 20 <= ADX <= 25
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_prev_4h[i] < -1.5) and (fisher_4h[i] >= -1.5) if not np.isnan(fisher_prev_4h[i]) else False
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_prev_4h[i] > 1.5) and (fisher_4h[i] <= 1.5) if not np.isnan(fisher_prev_4h[i]) else False
        
        # Fisher extreme levels (for mean reversion)
        fisher_extreme_oversold = fisher_4h[i] < -2.0
        fisher_extreme_overbought = fisher_4h[i] > 2.0
        
        # === DI+ / DI- CROSSOVER ===
        di_bullish = di_plus_4h[i] > di_minus_4h[i]
        di_bearish = di_plus_4h[i] < di_minus_4h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === STRONG TREND REGIME (ADX > 25) — Trend Following ===
        if strong_trend:
            # Long: Bullish alignment + Fisher confirmation or DI+ cross
            if macro_bull or trend_12h_bullish:
                if kama_bullish and di_bullish:
                    desired_signal = BASE_SIZE
                elif fisher_long_cross and above_sma50:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish alignment + Fisher confirmation or DI- cross
            if macro_bear or trend_12h_bearish:
                if kama_bearish and di_bearish:
                    desired_signal = -BASE_SIZE
                elif fisher_short_cross and below_sma50:
                    desired_signal = -REDUCED_SIZE
        
        # === WEAK TREND / RANGING REGIME (ADX < 20) — Mean Reversion ===
        elif weak_trend:
            # Long: Fisher extreme oversold + macro support
            if fisher_extreme_oversold and (macro_bull or above_sma200):
                desired_signal = REDUCED_SIZE
            
            # Short: Fisher extreme overbought + macro resistance
            if fisher_extreme_overbought and (macro_bear or below_sma200):
                desired_signal = -REDUCED_SIZE
            
            # Fallback: KAMA mean reversion
            if kama_bullish and fisher_4h[i] < -1.0 and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if kama_bearish and fisher_4h[i] > 1.0 and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) — Conservative ===
        else:
            # Require more confluence
            if macro_bull and trend_12h_bullish and kama_bullish:
                if fisher_long_cross or fisher_4h[i] < -1.0:
                    desired_signal = REDUCED_SIZE
            
            if macro_bear and trend_12h_bearish and kama_bearish:
                if fisher_short_cross or fisher_4h[i] > 1.0:
                    desired_signal = -REDUCED_SIZE
            
            # Fallback: Fisher extreme alone
            if fisher_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if fisher_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and Fisher not overbought
                if (macro_bull or trend_12h_bullish) and fisher_4h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and Fisher not oversold
                if (macro_bear or trend_12h_bearish) and fisher_4h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + medium trend reverses + Fisher overbought
            if macro_bear and trend_12h_bearish and fisher_4h[i] > 1.5:
                desired_signal = 0.0
            # Exit if ADX becomes very weak (trend dying)
            if adx_4h[i] < 15 and kama_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + medium trend reverses + Fisher oversold
            if macro_bull and trend_12h_bullish and fisher_4h[i] < -1.5:
                desired_signal = 0.0
            # Exit if ADX becomes very weak (trend dying)
            if adx_4h[i] < 15 and kama_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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