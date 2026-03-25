#!/usr/bin/env python3
"""
Experiment #1615: 6h Primary + 12h/1d HTF — Supertrend + ADX Regime + Stelzer RSI

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). 
Using Supertrend for clear directional bias + ADX for regime detection + 
Stelzer RSI (modified RSI with better extreme detection) should capture 
trends while avoiding whipsaws that killed previous 6h strategies.

Key innovations vs failed 6h attempts:
1. SUPERTREND (ATR=10, mult=3): Clear trend direction, avoids EMA whipsaw
2. ADX(14) regime: >25 = trend (follow Supertrend), <20 = range (mean revert)
3. STELZER RSI: RSI(7) with 3-bar smoothing, better at catching extremes
4. 12h HMA(21) for intermediate trend confirmation
5. 1d HMA(21) for long-term bias (prevents major counter-trend)
6. Asymmetric entries: trend regime uses Supertrend flip, range uses RSI extremes
7. Conservative sizing: 0.20 base, 0.30 strong signals

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- Supertrend proven superior to EMA/HMA for trend direction (less whipsaw)
- ADX regime filter prevents trend-following in choppy markets
- Stelzer RSI more responsive than standard RSI for entry timing
- 6h TF = better entry timing than 12h, fewer trades than 4h

Entry logic:
- LONG trend: ADX>25 + Supertrend=long + 12h_HMA bullish + 1d_HMA bullish + Stelzer_RSI<40 pullback
- SHORT trend: ADX>25 + Supertrend=short + 12h_HMA bearish + 1d_HMA bearish + Stelzer_RSI>60 pullback
- LONG range: ADX<20 + Supertrend=long + Stelzer_RSI<25 extreme
- SHORT range: ADX<20 + Supertrend=short + Stelzer_RSI>75 extreme

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_supertrend_adx_regime_stelzer_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - ATR-based trend following
    Returns: supertrend_values, direction (1=long, -1=short)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[period] = upper_band[period]
    direction[period] = -1  # Start short
    
    for i in range(period + 1, n):
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
        
        # Supertrend logic
        if direction[i-1] == 1:  # Previously long
            if close[i] > lower_band[i]:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:  # Previously short
            if close[i] < upper_band[i]:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di_pct = np.zeros(n, dtype=np.float64)
    minus_di_pct = np.zeros(n, dtype=np.float64)
    
    mask = atr_smooth > 1e-10
    plus_di_pct[mask] = 100.0 * plus_di[mask] / atr_smooth[mask]
    minus_di_pct[mask] = 100.0 * minus_di[mask] / atr_smooth[mask]
    
    # Calculate DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    di_sum = plus_di_pct + minus_di_pct
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di_pct[mask2] - minus_di_pct[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_stelzer_rsi(close, period=7, smooth=3):
    """
    Stelzer RSI - Modified RSI with smoothing for better extreme detection
    Uses shorter period (7) with 3-bar SMA smoothing
    """
    n = len(close)
    if n < period + smooth:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi_raw = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi_raw[mask] = 100 - (100 / (1 + rs[mask]))
    
    # Apply 3-bar SMA smoothing
    rsi_smooth = pd.Series(rsi_raw).rolling(window=smooth, min_periods=smooth).mean().values
    
    return rsi_smooth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_10 = calculate_atr(high, low, close, period=10)
    supertrend, supertrend_dir = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    adx_14 = calculate_adx(high, low, close, period=14)
    stelzer_rsi = calculate_stelzer_rsi(close, period=7, smooth=3)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Supertrend flips
    prev_supertrend_dir = np.nan
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_10[i]) or atr_10[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(supertrend_dir[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(stelzer_rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX) ===
        adx = adx_14[i]
        is_trend_regime = adx > 25.0
        is_range_regime = adx < 20.0
        
        # === TREND DIRECTION (Supertrend + HTF HMA bias) ===
        supertrend_long = supertrend_dir[i] == 1
        supertrend_short = supertrend_dir[i] == -1
        
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === STELZER RSI SIGNALS ===
        rsi_val = stelzer_rsi[i]
        rsi_oversold = rsi_val < 40.0
        rsi_overbought = rsi_val > 60.0
        rsi_extreme_low = rsi_val < 25.0
        rsi_extreme_high = rsi_val > 75.0
        
        # === SUPERSTREND FLIP DETECTION ===
        supertrend_flip_long = False
        supertrend_flip_short = False
        
        if not np.isnan(prev_supertrend_dir):
            if prev_supertrend_dir == -1 and supertrend_dir[i] == 1:
                supertrend_flip_long = True
            elif prev_supertrend_dir == 1 and supertrend_dir[i] == -1:
                supertrend_flip_short = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow Supertrend with HTF confirmation + RSI pullback
        if is_trend_regime:
            # LONG: Supertrend long + 12h/1d bullish + RSI pullback (not overbought)
            if supertrend_long and price_above_12h and price_above_1d and rsi_oversold:
                desired_signal = SIZE_STRONG if supertrend_flip_long else SIZE_BASE
            
            # SHORT: Supertrend short + 12h/1d bearish + RSI pullback (not oversold)
            elif supertrend_short and price_below_12h and price_below_1d and rsi_overbought:
                desired_signal = -SIZE_STRONG if supertrend_flip_short else -SIZE_BASE
        
        # RANGE REGIME: Mean reversion with Supertrend direction + RSI extremes
        elif is_range_regime:
            # LONG: Supertrend long + RSI extreme low
            if supertrend_long and rsi_extreme_low:
                desired_signal = SIZE_BASE
            
            # SHORT: Supertrend short + RSI extreme high
            elif supertrend_short and rsi_extreme_high:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME (ADX 20-25): Use HTF bias only
        else:
            # LONG: 1d bullish + Supertrend long + RSI not extreme high
            if price_above_1d and supertrend_long and rsi_val < 70.0:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + Supertrend short + RSI not extreme low
            elif price_below_1d and supertrend_short and rsi_val > 30.0:
                desired_signal = -SIZE_BASE
        
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
                entry_atr = atr_10[i]
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
        prev_supertrend_dir = supertrend_dir[i]
    
    return signals