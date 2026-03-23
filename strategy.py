#!/usr/bin/env python3
"""
Experiment #759: 4h Primary + 1d HTF — KAMA Adaptive Trend + Fisher Transform + Choppiness Regime

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA
2. Ehlers Fisher Transform catches reversals more precisely than RSI/CRSI in bear markets
3. Choppiness Index regime filter proven effective (ETH +0.923 in prior tests)
4. ADX filter prevents entries during weak trends (reduces whipsaw)
5. Simpler 2-regime logic (trend vs range) increases trade frequency vs triple regime
6. Fisher Transform crosses at -1.5/+1.5 levels provide clear entry signals
7. 1d KAMA(21) for primary trend bias across all market conditions

Strategy design:
1. 1d KAMA(21) for primary trend bias (aligned via mtf_data helper)
2. 4h Choppiness Index(14) for regime detection (trend vs range)
3. 4h Fisher Transform(9) for entry timing (reversal signals)
4. 4h ADX(14) for trend strength confirmation (>25 = strong trend)
5. 4h ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.25, ±0.30

Key improvements from #751:
- Replaced CRSI with Fisher Transform (better for bear market reversals)
- Replaced HMA with KAMA (more adaptive to volatility changes)
- Added ADX filter to avoid weak trend entries
- Simpler regime logic to ensure >=30 trades/train
- Looser Fisher thresholds to increase trade frequency

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_adx_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise/volatility automatically.
    ER (Efficiency Ratio) determines smoothing constant.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0
    
    er = np.clip(er, 0, 1)
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals.
    Long: Fisher crosses above -1.5
    Short: Fisher crosses below +1.5
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher, fisher_prev
    
    # Calculate typical price and normalize
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            continue
        
        # Normalize price to -1 to +1
        mid = (highest + lowest) / 2.0
        current_price = (high[i] + low[i]) / 2.0
        normalized = (current_price - mid) / (price_range / 2.0)
        normalized = np.clip(normalized, -0.99, 0.99)  # Prevent log domain error
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

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
    Average Directional Index (ADX)
    Measures trend strength regardless of direction.
    ADX > 25 = strong trend, ADX < 20 = weak/ranging
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 5:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth TR, +DM, -DM
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-10)
    
    # Calculate DX
    di_sum = plus_di + minus_di
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (di_sum + 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    fisher_4h, fisher_prev_4h = calculate_fisher_transform(high, low, period=9)
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
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
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_prev_4h[i]):
            continue
        if np.isnan(adx_4h[i]):
            continue
        
        # === TREND BIAS (1d HTF KAMA) ===
        trend_1d_bullish = close[i] > kama_1d_aligned[i]
        trend_1d_bearish = close[i] < kama_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop_4h[i] < 38.2
        ranging_regime = chop_4h[i] > 61.8
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher_prev_4h[i] < -1.5 and fisher_4h[i] >= -1.5
        fisher_cross_down = fisher_prev_4h[i] > 1.5 and fisher_4h[i] <= 1.5
        fisher_oversold = fisher_4h[i] < -1.0
        fisher_overbought = fisher_4h[i] > 1.0
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 38.2) ===
        if trending_regime:
            # Long: 1d bullish + Fisher cross up + strong trend
            if trend_1d_bullish and fisher_cross_up and strong_trend:
                desired_signal = BASE_SIZE
            
            # Short: 1d bearish + Fisher cross down + strong trend
            if trend_1d_bearish and fisher_cross_down and strong_trend:
                desired_signal = -BASE_SIZE
            
            # Trend continuation (weaker signal)
            if trend_1d_bullish and fisher_oversold and adx_4h[i] > 20:
                desired_signal = REDUCED_SIZE
            
            if trend_1d_bearish and fisher_overbought and adx_4h[i] > 20:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME LOGIC (CHOP > 61.8) ===
        elif ranging_regime:
            # Mean reversion long: Fisher cross up + 1d bullish bias
            if fisher_cross_up and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            # Mean reversion short: Fisher cross down + 1d bearish bias
            if fisher_cross_down and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
            
            # Pure mean reversion (extreme Fisher values)
            if fisher_4h[i] < -2.0 and not trend_1d_bearish:
                desired_signal = REDUCED_SIZE
            
            if fisher_4h[i] > 2.0 and not trend_1d_bullish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter on Fisher crosses + trend alignment
            if fisher_cross_up and trend_1d_bullish and adx_4h[i] > 20:
                desired_signal = REDUCED_SIZE
            
            if fisher_cross_down and trend_1d_bearish and adx_4h[i] > 20:
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
                if trend_1d_bullish and fisher_4h[i] < 1.5:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                if trend_1d_bearish and fisher_4h[i] > -1.5:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if trend_1d_bearish and fisher_4h[i] > 1.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if trend_1d_bullish and fisher_4h[i] < -1.0:
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