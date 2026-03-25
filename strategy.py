#!/usr/bin/env python3
"""
Experiment #1375: 6h Primary + 12h/1d HTF — Fisher Transform + ADX Regime + Vol Spike

Hypothesis: 6h timeframe needs regime-adaptive logic (trend-following failed in 2022 crash).
This strategy combines:
1. Fisher Transform (period=9) - catches reversals in bear markets (75% win rate per research)
2. ADX(14) regime detection - trending (ADX>22) vs ranging (ADX<20) with hysteresis
3. 12h HMA(21) for intermediate trend bias (smoother than 1d alone)
4. 1d HMA(21) for major trend bias
5. ATR vol spike detection - enter mean reversion when vol spikes (ATR7/ATR30 > 1.8)
6. Asymmetric logic - different entry rules for bull/bear/range regimes

Why this should work where others failed:
- Fisher Transform excels at catching bear market reversals (research-backed)
- ADX regime filter prevents trend-following in ranges (where 2022 whipsaw occurred)
- Vol spike reversion captures "panic bottoms" common in crypto crashes
- 12h HTF gives smoother trend signal than 1d alone for 6h entries
- LOOSE entry thresholds guarantee trades (common failure: 0 trades)

Entry logic:
- TRENDING (ADX>22): Fisher crosses -1.5 + price>12h_HMA → long
- RANGING (ADX<20): Fisher extreme + BB touch → mean revert
- VOL SPIKE: ATR7/ATR30>1.8 + price<BB_lower → long (panic bottom)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_regime_vol_spike_12h1d_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals better than RSI in bear markets
    """
    n = len(high)
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        if highest > lowest:
            # Normalize to 0-1 range
            normalized = (hl2 - lowest) / (highest - lowest)
            
            # Clamp to avoid log(0) or log(inf)
            normalized = max(0.001, min(0.999, normalized))
            
            # Fisher calculation
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            
            # Smooth with previous value
            if i > period and not np.isnan(fisher[i-1]):
                fisher[i] = 0.7 * fisher[i] + 0.3 * fisher_prev[i-1]
            
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength indicator"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0.0
        
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0.0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    mask = tr_smooth > 0
    plus_di[mask] = (plus_dm_smooth[mask] / tr_smooth[mask]) * 100
    minus_di[mask] = (minus_dm_smooth[mask] / tr_smooth[mask]) * 100
    
    # Calculate DX and ADX
    dx = np.full(n, np.nan, dtype=np.float64)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2]) * 100
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for vol spike detection"""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(len(close), np.nan, dtype=np.float64)
    mask = (atr_long > 0) & (~np.isnan(atr_short)) & (~np.isnan(atr_long))
    ratio[mask] = atr_short[mask] / atr_long[mask]
    
    return ratio

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with bandwidth calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

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
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    adx_14 = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Discrete position sizing (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.15
    
    # Position tracking for stoploss (Rule 6)
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    prev_fisher = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(adx_14[i]):
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
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX with hysteresis) ===
        adx = adx_14[i]
        is_trending = adx > 22  # Trending regime
        is_ranging = adx < 20   # Ranging regime (hysteresis gap)
        
        # === TREND DIRECTION (12h HMA bias) ===
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # 1d HMA for major regime filter
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev_val = fisher_prev[i] if not np.isnan(fisher_prev[i]) else fisher_val
        
        # Fisher crossover detection
        fisher_cross_up = (fisher_prev_val < -1.5) and (fisher_val >= -1.5)
        fisher_cross_down = (fisher_prev_val > 1.5) and (fisher_val <= 1.5)
        fisher_extreme_low = fisher_val < -1.8
        fisher_extreme_high = fisher_val > 1.8
        
        # === VOL SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.8  # Volatility 80% above normal
        
        # === PRICE POSITION ===
        price_near_bb_lower = close[i] <= bb_lower[i] * 1.005  # Within 0.5% of lower BB
        price_near_bb_upper = close[i] >= bb_upper[i] * 0.995  # Within 0.5% of upper BB
        
        rsi = rsi_14[i] if not np.isnan(rsi_14[i]) else 50
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # SCENARIO 1: VOL SPIKE MEAN REVERSION (highest priority - panic bottoms)
        if vol_spike and price_near_bb_lower and rsi < 45:
            # Panic selling - buy the dip
            if price_above_1d:
                base_size = SIZE_STRONG  # Strong conviction with 1d trend
            else:
                base_size = SIZE_BASE
            desired_signal = base_size
        
        elif vol_spike and price_near_bb_upper and rsi > 55:
            # Panic buying - sell the rip
            if price_below_1d:
                base_size = SIZE_STRONG
            else:
                base_size = SIZE_BASE
            desired_signal = -base_size
        
        # SCENARIO 2: TRENDING REGIME (ADX > 22)
        elif is_trending:
            if price_above_12h and fisher_cross_up:
                # Trending long - Fisher cross above -1.5
                if price_above_1d:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            elif price_below_12h and fisher_cross_down:
                # Trending short - Fisher cross below +1.5
                if price_below_1d:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # SCENARIO 3: RANGING REGIME (ADX < 20)
        elif is_ranging:
            if fisher_extreme_low and price_near_bb_lower:
                # Range bottom - mean revert long
                desired_signal = SIZE_BASE
            
            elif fisher_extreme_high and price_near_bb_upper:
                # Range top - mean revert short
                desired_signal = -SIZE_BASE
        
        # SCENARIO 4: TRANSITION REGIME (20 <= ADX <= 22) - use Fisher extremes only
        else:
            if fisher_extreme_low and rsi < 40:
                desired_signal = SIZE_WEAK
            elif fisher_extreme_high and rsi > 60:
                desired_signal = -SIZE_WEAK
        
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
        
        # === DISCRETIZE SIGNAL VALUES (Rule 4) ===
        if desired_signal >= SIZE_STRONG * 0.8:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.8:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.8:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.8:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_WEAK * 0.8:
            final_signal = SIZE_WEAK if desired_signal > 0 else -SIZE_WEAK
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
        prev_fisher = fisher_val
    
    return signals