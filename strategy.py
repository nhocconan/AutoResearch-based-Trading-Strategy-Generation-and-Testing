#!/usr/bin/env python3
"""
Experiment #754: 4h Primary + 12h HTF — KAMA Adaptive Trend + Fisher Transform + Choppiness Regime

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA - reduces whipsaw in chop
2. Ehlers Fisher Transform catches reversals in bear rallies (research notes show success in 2022 crash)
3. Choppiness Index regime filter proven working (ETH +0.923 in prior tests)
4. 12h HTF trend bias (not 1d) may capture intermediate trends better for 4h entries
5. ADX filter ensures we only trade when momentum exists (avoid dead zones)
6. Looser Fisher thresholds (-1.8/+1.8) ensure >=30 trades/train while maintaining quality

Strategy design:
1. 12h KAMA(21) for adaptive trend bias (aligned via mtf_data helper)
2. 4h Choppiness Index(14) for regime detection (trend vs range)
3. 4h Fisher Transform(9) for entry timing (crosses -1.5/+1.5)
4. 4h ADX(14) for trend strength confirmation (>20 for trend entries)
5. 4h ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.25, ±0.30

Key differences from #751:
- KAMA instead of HMA (adaptive to volatility)
- Fisher Transform instead of Connors RSI (better for reversals)
- 12h HTF instead of 1d (intermediate trend capture)
- ADX filter for momentum confirmation
- Different entry logic for bear/range markets

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_12h_adx_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    More responsive in trends, smoother in chop.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(slow_period, n):
        if np.isnan(close[i]) or np.isnan(close[i-slow_period]):
            continue
        price_change = np.abs(close[i] - close[i-slow_period])
        volatility = np.sum(np.abs(np.diff(close[i-slow_period:i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    er = np.clip(er, 0, 1)
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc ** 2
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    # Calculate KAMA
    for i in range(slow_period + 1, n):
        if np.isnan(kama[i-1]) or np.isnan(close[i]):
            continue
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for clearer signals.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher, fisher_signal
    
    # Calculate typical price
    typical = (high + low + close) / 3
    
    # Normalize price to -1 to +1 range
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    normalized = np.zeros(n)
    for i in range(period, n):
        if np.isnan(highest[i]) or np.isnan(lowest[i]):
            continue
        price_range = highest[i] - lowest[i]
        if price_range > 1e-10:
            normalized[i] = 0.999 * (2 * (typical[i] - lowest[i]) / price_range - 1)
        else:
            normalized[i] = 0
    
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Calculate Fisher Transform
    for i in range(period, n):
        if np.isnan(normalized[i]):
            continue
        # Fisher = 0.5 * ln((1+normalized)/(1-normalized))
        with np.errstate(divide='ignore', invalid='ignore'):
            fisher[i] = 0.5 * np.log((1 + normalized[i]) / (1 - normalized[i]) + 1e-10)
    
    # Fisher Signal (previous bar Fisher)
    fisher_signal[period+1:] = fisher[period:-1]
    
    return fisher, fisher_signal

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index.
    Measures trend strength (not direction).
    ADX > 25 = strong trend, ADX < 20 = weak/ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    if n < period * 2 + 5:
        return adx, plus_di, minus_di
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR and DM
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-10)
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures whether market is trending or ranging.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, close, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high, low, close, period=14)
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    
    # Calculate and align HTF KAMA for trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(chop_4h[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(plus_di_4h[i]) or np.isnan(minus_di_4h[i]):
            continue
        
        # === TREND BIAS (12h HTF KAMA) ===
        trend_12h_bullish = close[i] > kama_12h_aligned[i]
        trend_12h_bearish = close[i] < kama_12h_aligned[i]
        
        # 4h KAMA trend
        trend_4h_bullish = close[i] > kama_4h[i]
        trend_4h_bearish = close[i] < kama_4h[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop_4h[i] < 38.2
        ranging_regime = chop_4h[i] > 61.8
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher_signal_4h[i] < -1.8 and fisher_4h[i] >= -1.8
        fisher_cross_down = fisher_signal_4h[i] > 1.8 and fisher_4h[i] <= 1.8
        fisher_oversold = fisher_4h[i] < -1.5
        fisher_overbought = fisher_4h[i] > 1.5
        
        # === DI CROSSOVER ===
        di_bullish = plus_di_4h[i] > minus_di_4h[i]
        di_bearish = plus_di_4h[i] < minus_di_4h[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2 + ADX > 25) ===
        if trending_regime and strong_trend:
            # Long: 12h bullish + 4h bullish + Fisher cross up + DI bullish
            if trend_12h_bullish and trend_4h_bullish and fisher_cross_up and di_bullish:
                desired_signal = BASE_SIZE
            
            # Short: 12h bearish + 4h bearish + Fisher cross down + DI bearish
            if trend_12h_bearish and trend_4h_bearish and fisher_cross_down and di_bearish:
                desired_signal = -BASE_SIZE
            
            # Trend continuation (looser entry)
            if trend_12h_bullish and trend_4h_bullish and fisher_oversold and di_bullish:
                desired_signal = BASE_SIZE
            
            if trend_12h_bearish and trend_4h_bearish and fisher_overbought and di_bearish:
                desired_signal = -BASE_SIZE
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) ===
        elif ranging_regime:
            # Mean reversion long: Fisher extreme low + 12h not bearish
            if fisher_4h[i] < -2.0 and not trend_12h_bearish:
                desired_signal = REDUCED_SIZE
            
            # Mean reversion short: Fisher extreme high + 12h not bullish
            if fisher_4h[i] > 2.0 and not trend_12h_bullish:
                desired_signal = -REDUCED_SIZE
            
            # Pure mean reversion with ADX confirmation
            if fisher_cross_up and weak_trend:
                desired_signal = REDUCED_SIZE
            
            if fisher_cross_down and weak_trend:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter on Fisher extremes + trend alignment
            if fisher_cross_up and trend_12h_bullish and di_bullish:
                desired_signal = REDUCED_SIZE
            
            if fisher_cross_down and trend_12h_bearish and di_bearish:
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 12h trend intact and Fisher not overbought
                if trend_12h_bullish and fisher_4h[i] < 1.5:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 12h trend intact and Fisher not oversold
                if trend_12h_bearish and fisher_4h[i] > -1.5:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 12h trend reverses or Fisher overbought
            if trend_12h_bearish and fisher_4h[i] > 1.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 12h trend reverses or Fisher oversold
            if trend_12h_bullish and fisher_4h[i] < -1.0:
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