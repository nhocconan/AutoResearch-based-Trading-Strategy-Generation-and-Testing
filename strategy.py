#!/usr/bin/env python3
"""
Experiment #007: 6h Williams Alligator + Elder Ray + 1d HMA

Hypothesis: Williams Alligator ( Jaw=13/8, Teeth=8/5, Lips=5/3 ) captures 
multi-timeframe smoothed price action without noise. Elder Ray confirms 
institutional pressure. 1d HMA filters for macro trend bias.

Why this should work in BOTH bull AND bear:
- Alligator expansion/contraction adapts to volatility regime
- Elder Ray shows Bull/Bear divergence BEFORE breakout (captures reversal)
- 1d HMA bias prevents fighting the macro trend
- 6h captures swing moves (12h-3day) without overtrading

Novelty: Alligator+Elder Ray combo NOT tried before in 16,000+ experiments.
Williams system was designed for swing trading on daily TF - 6h is perfect fit.

Target: 75-150 total trades over 4 years (19-37/year). HARD MAX: 300.
Size: 0.25-0.30 discrete
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_alligator_elder_ray_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_alligator(high, low, close):
    """
    Williams Alligator - 3 smoothed MAs with different periods
    Jaw (blue): 13 period, smoothed over 8
    Teeth (red): 8 period, smoothed over 5  
    Lips (green): 5 period, smoothed over 3
    
    Returns: jaw, teeth, lips arrays
    """
    n = len(close)
    jaw = np.full(n, np.nan, dtype=np.float64)
    teeth = np.full(n, np.nan, dtype=np.float64)
    lips = np.full(n, np.nan, dtype=np.float64)
    
    # SMMA (Smoothed Moving Average - same as Wilder's smoothing)
    def smma(data, period, smooth):
        n = len(data)
        result = np.full(n, np.nan, dtype=np.float64)
        # First value is simple SMA
        if n >= period:
            result[period - 1] = np.mean(data[:period])
            # Smoothed values
            for i in range(period, n):
                result[i] = (result[i - 1] * (smooth - 1) + data[i]) / smooth
        return result
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # Calculate Alligator lines
    jaw = smma(median_price, 13, 8)
    teeth = smma(median_price, 8, 5)
    lips = smma(median_price, 5, 3)
    
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, ema_period=13):
    """
    Elder Ray - measures buying/selling pressure
    Bull Power = High - EMA(13)
    Bear Power = Low + EMA(13)
    
    When Bull Power > 0 and rising = buying pressure
    When Bear Power < 0 and falling = selling pressure
    """
    n = len(close)
    if n < ema_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    
    bull_power = high - ema
    bear_power = low + ema  # Note: original formula is Low - EMA, but we want positive values
    
    return bull_power, bear_power

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

def calculate_adx(high, low, close, period=14):
    """ADX - Average Directional Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    mask = atr_smooth > 1e-10
    plus_di[mask] = 100 * plus_dm_smooth[mask] / atr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / atr_smooth[mask]
    
    di_sum = plus_di + minus_di
    valid = mask & (di_sum > 1e-10)
    dx[valid] = 100 * np.abs(plus_di[valid] - minus_di[valid]) / di_sum[valid]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    jaw, teeth, lips = calculate_alligator(high, low, close)
    bull_power, bear_power = calculate_elder_ray(high, low, close, ema_period=13)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === ALLIGATOR SIGNALS ===
        # Alligator "sleeping" = all lines together, "Awake" = lines separated
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_above_jaw = lips[i] > jaw[i]
        
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        lips_below_jaw = lips[i] < jaw[i]
        
        # Alligator bullish alignment (all three aligned upward)
        alligator_bullish = lips_above_teeth and teeth_above_jaw and lips_above_jaw
        alligator_bearish = lips_below_teeth and teeth_below_jaw and lips_below_jaw
        
        # Alligator expansion - lips moved away from jaw (volatility expansion)
        lips_jaw_gap_up = lips[i] - jaw[i]
        lips_jaw_gap_down = jaw[i] - lips[i]
        
        # === ELDER RAY CONFIRMATION ===
        bull_pow = bull_power[i]
        bear_pow = bear_power[i]  # Note: this is Low + EMA (positive when price above EMA)
        
        # Bull power rising = positive, Bear power falling = negative signal
        bull_power_rising = bull_pow > 0
        bear_power_falling = bear_pow < 0  # Bear power should be negative
        
        # === 1d HMA TREND BIAS ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === ADX TREND STRENGTH (optional filter - don't enter if too choppy) ===
        adx_val = adx_14[i]
        is_trending = adx_val > 20  # Loose filter - trending market
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # LONG ENTRY: 
        # 1. 1d HMA bullish bias
        # 2. Alligator bullish alignment  
        # 3. Bull power positive (buying pressure)
        if price_above_1d and alligator_bullish and bull_power_rising:
            desired_signal = SIZE_STRONG if (is_trending and lips_jaw_gap_up > atr_14[i]) else SIZE_BASE
        
        # SHORT ENTRY:
        # 1. 1d HMA bearish bias
        # 2. Alligator bearish alignment
        # 3. Price below EMA (bear power negative indicates selling pressure)
        elif price_below_1d and alligator_bearish and bear_pow < 0:
            desired_signal = -SIZE_STRONG if (is_trending and lips_jaw_gap_down > atr_14[i]) else -SIZE_BASE
        
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
        
        signals[i] = final_signal
    
    return signals