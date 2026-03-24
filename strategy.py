#!/usr/bin/env python3
"""
Experiment #920: 6h Primary + 1d/1w HTF — Fisher Transform + ADX Regime + Weekly HMA

Hypothesis: 6h timeframe fills gap between 4h (too many trades) and 12h (too few).
Fisher Transform catches reversals better than RSI in bear/range markets (proven in 2022 crash).
Daily ADX regime filter adapts logic: trending (follow Fisher with bias) vs ranging (mean revert).
Weekly HMA provides ultra-long-term bias to avoid counter-trend trades.
Choppiness Index confirms regime and avoids false signals in chop.

Key innovations:
1. 1w HMA(21) for ultra-long-term bias - price above = bullish bias only
2. 1d ADX(14) for regime detection - ADX>25 trending, ADX<20 ranging
3. 6h Fisher Transform(9) for entry timing - crosses -1.5 long, +1.5 short
4. 6h Choppiness Index(14) confirms regime - CHOP<38 trend, CHOP>62 range
5. Regime-adaptive logic: trend=follow Fisher+bias, range=mean revert extremes
6. ATR(14) 2.5x trailing stop for risk management
7. LOOSE entry conditions to ensure ≥30 trades/train, ≥5/test

Entry conditions (LOOSE to guarantee trades):
- TREND REGIME (ADX>25): Fisher cross with weekly bias confirmation
- RANGE REGIME (ADX<20): Fisher extreme reversal opposite to weekly bias
- TRANSITION (ADX 20-25): reduced size, Fisher cross only

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Fisher = 0.5 * ln((1+X)/(1-X)) where X = EMA(price, period) normalized
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize to -1 to +1 range using highest high and lowest low over period
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest - lowest < 1e-10:
            continue
        
        # Normalize price to -1 to +1
        x = 2.0 * (typical[i] - lowest) / (highest - lowest) - 1.0
        
        # Clamp to avoid log errors
        x = max(-0.999, min(0.999, x))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Trigger line (1-period lag of Fisher)
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength regardless of direction
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smooth with Wilder's method (EMA with alpha=1/period)
    atr = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    atr[period-1] = np.sum(tr[:period]) / period
    plus_di[period-1] = 100.0 * np.sum(plus_dm[:period]) / atr[period-1] if atr[period-1] > 1e-10 else 0.0
    minus_di[period-1] = 100.0 * np.sum(minus_dm[:period]) / atr[period-1] if atr[period-1] > 1e-10 else 0.0
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100.0 * ((plus_di[i-1] * atr[i-1] * (period - 1) / 100.0) + plus_dm[i]) / atr[i] if atr[i] > 1e-10 else 0.0
        minus_di[i] = 100.0 * ((minus_di[i-1] * atr[i-1] * (period - 1) / 100.0) + minus_dm[i]) / atr[i] if atr[i] > 1e-10 else 0.0
    
    # DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    adx = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # ADX = SMA of DX over period
    for i in range(period * 2 - 1, n):
        adx[i] = np.mean(dx[i-period+1:i+1])
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest - lowest < 1e-10:
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate 6h indicators
    fisher_6h, trigger_6h = calculate_fisher_transform(high, low, period=9)
    chop_6h = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher_6h[i]) or np.isnan(trigger_6h[i]) or np.isnan(chop_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME FILTER (1d ADX) ===
        adx_value = adx_1d_aligned[i]
        regime_trending = adx_value > 25.0
        regime_ranging = adx_value < 20.0
        regime_transition = not regime_trending and not regime_ranging
        
        # === CHOPPINESS CONFIRMATION (6h CHOP) ===
        chop_trending = chop_6h[i] < 38.2
        chop_ranging = chop_6h[i] > 61.8
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 0 and not np.isnan(fisher_6h[i-1]) and not np.isnan(trigger_6h[i-1]):
            # Fisher crosses above trigger (bullish)
            fisher_cross_long = (fisher_6h[i-1] <= trigger_6h[i-1]) and (fisher_6h[i] > trigger_6h[i])
            # Fisher crosses below trigger (bearish)
            fisher_cross_short = (fisher_6h[i-1] >= trigger_6h[i-1]) and (fisher_6h[i] < trigger_6h[i])
        
        # Fisher extreme levels for mean reversion
        fisher_oversold = fisher_6h[i] < -1.5
        fisher_overbought = fisher_6h[i] > 1.5
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE, LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow Fisher with weekly bias
        if regime_trending or chop_trending:
            # Long: Fisher cross up + weekly bullish OR just Fisher cross up (loose)
            if fisher_cross_long:
                if htf_1w_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE  # loose: allow counter-trend in strong trend
            
            # Short: Fisher cross down + weekly bearish OR just Fisher cross down (loose)
            elif fisher_cross_short:
                if htf_1w_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE  # loose: allow counter-trend in strong trend
        
        # RANGING REGIME: Mean revert at Fisher extremes
        elif regime_ranging or chop_ranging:
            # Long: Fisher oversold (mean reversion)
            if fisher_oversold:
                desired_signal = SIZE_BASE
            
            # Short: Fisher overbought (mean reversion)
            elif fisher_overbought:
                desired_signal = -SIZE_BASE
        
        # TRANSITION REGIME: Reduced size, Fisher cross only
        elif regime_transition:
            if fisher_cross_long:
                desired_signal = SIZE_BASE * 0.8
            elif fisher_cross_short:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals