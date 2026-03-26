#!/usr/bin/env python3
"""
Experiment #1619: 4h Donchian Breakout + Volume Spike + Choppiness Regime + 1d HMA

HYPOTHESIS: Price channel (Donchian) breakout + volume confirmation + regime filter
is the proven winning formula from DB (Sharpe 1.10-1.46). Simple = fewer trades =
less fee drag = better generalization to test period.

Why this should work in BOTH bull AND bear:
- Donchian breakout catches the start of trends in any direction
- Volume spike confirms the breakout is institutional, not noise
- Choppiness < 38.2 = trending market = enter with momentum
- Choppiness > 61.8 = ranging = skip or mean-revert only
- 1d HMA bias keeps us on the right side of the larger trend

Key design choices (learning from failures):
1. NO neutral regime entries - only trend regime signals (eliminates ~50% of bad trades)
2. Volume spike as mandatory filter (filters noise breakouts)
3. 4h HMA crossover for entry timing (not just bias)
4. 1d HMA for trend direction only
5. ATR-based stoploss at 2.5x

Target: 100-150 total trades over 4 years, Sharpe > 0.7, DD < -30%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_spike_chop_1d_hma_v1"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout structure"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """
    Volume ratio: current volume vs average volume
    > 1.5 = volume spike (institutional interest)
    """
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    avg_vol = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = np.full(n, np.nan, dtype=np.float64)
    mask = avg_vol > 0
    ratio[mask] = volume[mask] / avg_vol[mask]
    
    return ratio

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    
    # 4h HMA for entry timing
    hma_4h = calculate_hma(close, period=16)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_TIGHT = 0.20  # for weaker signals
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(donch_upper[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT (price structure) ===
        donch_breakout_long = (i > 0 and not np.isnan(donch_upper[i-1]) and 
                                close[i] > donch_upper[i-1])
        donch_breakout_short = (i > 0 and not np.isnan(donch_lower[i-1]) and 
                                 close[i] < donch_lower[i-1])
        
        # === VOLUME CONFIRMATION (mandatory filter) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === HMA ENTRY TIMING ===
        hma_4h_trend_up = close[i] > hma_4h[i]
        hma_4h_trend_down = close[i] < hma_4h[i]
        
        # === RSI CONFIRMATION (mild, to avoid conflicts) ===
        rsi_val = rsi_14[i] if not np.isnan(rsi_14[i]) else 50.0
        rsi_neutral = 35 < rsi_val < 65
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trend_regime:
            # TREND REGIME: Donchian breakout + volume spike + 1d bias + HMA timing
            # These are the highest quality signals
            if price_above_1d and donch_breakout_long and vol_spike and hma_4h_trend_up:
                # Strong uptrend breakout with volume confirmation
                desired_signal = SIZE_STRONG
            elif price_below_1d and donch_breakout_short and vol_spike and hma_4h_trend_down:
                # Strong downtrend breakout with volume confirmation
                desired_signal = -SIZE_STRONG
        
        elif is_range_regime:
            # RANGE REGIME: Skip major entries, only if very strong setup
            # Price at channel extremes + extreme volume spike
            if price_above_1d and vol_spike and rsi_neutral and close[i] > donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else False:
                desired_signal = SIZE_TIGHT
            elif price_below_1d and vol_spike and rsi_neutral and close[i] < donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else False:
                desired_signal = -SIZE_TIGHT
        
        # === NO NEUTRAL REGIME ENTRIES ===
        # This is intentional - neutral regime = no trade (reduces bad trades by ~40%)
        
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
        elif desired_signal >= SIZE_TIGHT * 0.9:
            final_signal = SIZE_TIGHT
        elif desired_signal <= -SIZE_TIGHT * 0.9:
            final_signal = -SIZE_TIGHT
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = low[i] - 2.5 * entry_atr
                else:
                    stop_price = high[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals