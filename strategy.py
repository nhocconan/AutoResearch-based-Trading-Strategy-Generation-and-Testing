#!/usr/bin/env python3
"""
Experiment #1000: 6h Primary + 1d/1w HTF — Keltner Channel + ADX + Triple HMA Alignment

Hypothesis: 6h timeframe with Keltner Channel breakouts, ADX trend strength filter,
and triple HMA alignment (6h/1d/1w) will capture multi-day swings with better
risk-adjusted returns than pure HMA or Donchian strategies.

Key innovations:
1. Keltner Channel (EMA20, ATR10, mult=2.0): volatility-adjusted breakout levels
2. ADX(14) > 20: minimum trend strength filter (looser than typical 25)
3. Triple HMA alignment: 6h HMA(16), 1d HMA(21), 1w HMA(50) all aligned
4. Volume confirmation: taker_buy_volume ratio > 0.55 for longs, < 0.45 for shorts
5. ATR(14) 2.5x trailing stop for risk management
6. Multiple entry paths to ensure 30+ trades/train

Why 6h should work:
- Captures 2-5 day swings (between 4h noise and 12h lag)
- Fewer trades than 4h = less fee drag
- More responsive than 12h = catches reversals faster
- Keltner adapts to volatility (wider in high vol, tighter in low vol)

Entry conditions (designed for trade frequency):
- LONG = 1w HMA bull + 1d HMA bull + 6h HMA bull + ADX>20 + price>Keltner_upper + volume_confirm
- SHORT = 1w HMA bear + 1d HMA bear + 6h HMA bear + ADX>20 + price<Keltner_lower + volume_confirm
- Relaxed ADX threshold (20 vs 25) and multiple volume paths for more trades

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_adx_triple_hma_vol_v1"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr_vals = calculate_atr(high, low, close, period)
    
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if not np.isnan(atr_vals[i]) and atr_vals[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr_vals[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr_vals[i]
    
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period * 2, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            sum_di = plus_di[i] + minus_di[i]
            if sum_di > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / sum_di
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_keltner(high, low, close, ema_period=20, atr_period=10, mult=2.0):
    """Keltner Channel - returns upper, middle, lower bands"""
    n = len(close)
    if n < ema_period + atr_period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(ema_period + atr_period - 1, n):
        if not np.isnan(ema[i]) and not np.isnan(atr[i]):
            upper[i] = ema[i] + mult * atr[i]
            lower[i] = ema[i] - mult * atr[i]
    
    return upper, ema, lower

def calculate_volume_ratio(prices):
    """Calculate taker buy volume ratio"""
    n = len(prices)
    ratio = np.full(n, np.nan, dtype=np.float64)
    
    if 'taker_buy_volume' in prices.columns and 'volume' in prices.columns:
        for i in range(n):
            vol = prices['volume'].values[i]
            buy_vol = prices['taker_buy_volume'].values[i]
            if vol > 1e-10:
                ratio[i] = buy_vol / vol
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=16)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    keltner_upper, keltner_mid, keltner_lower = calculate_keltner(high, low, close, ema_period=20, atr_period=10, mult=2.0)
    volume_ratio = calculate_volume_ratio(prices)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TRIPLE HMA ALIGNMENT (6h + 1d + 1w) ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # All three aligned bull or all three aligned bear
        triple_bull = hma_6h_bull and hma_1d_bull and hma_1w_bull
        triple_bear = hma_6h_bear and hma_1d_bear and hma_1w_bear
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20.0  # Relaxed from 25 to ensure trades
        
        # === KELTNER BREAKOUT ===
        keltner_breakout_long = close[i] > keltner_upper[i-1] if i > 0 else False
        keltner_breakdown_short = close[i] < keltner_lower[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm_long = volume_ratio[i] > 0.52 if not np.isnan(volume_ratio[i]) else True
        vol_confirm_short = volume_ratio[i] < 0.48 if not np.isnan(volume_ratio[i]) else True
        
        # === ENTRY LOGIC (MULTIPLE PATHS FOR TRADE FREQUENCY) ===
        desired_signal = 0.0
        
        # LONG entries - multiple paths
        if triple_bull:
            # Path 1: Keltner breakout with ADX and volume
            if keltner_breakout_long and adx_strong and vol_confirm_long:
                desired_signal = SIZE_STRONG
            # Path 2: Price above Keltner mid + ADX strong (trend continuation)
            elif close[i] > keltner_mid[i] and adx_strong and hma_6h_bull:
                desired_signal = SIZE_BASE
            # Path 3: Pullback to Keltner mid in strong trend
            elif close[i] > keltner_lower[i] and close[i] < keltner_mid[i] and adx_strong > 25:
                desired_signal = SIZE_BASE
        
        # SHORT entries - multiple paths
        elif triple_bear:
            # Path 1: Keltner breakdown with ADX and volume
            if keltner_breakdown_short and adx_strong and vol_confirm_short:
                desired_signal = -SIZE_STRONG
            # Path 2: Price below Keltner mid + ADX strong (trend continuation)
            elif close[i] < keltner_mid[i] and adx_strong and hma_6h_bear:
                desired_signal = -SIZE_BASE
            # Path 3: Pullback to Keltner mid in strong trend
            elif close[i] < keltner_upper[i] and close[i] > keltner_mid[i] and adx_strong > 25:
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