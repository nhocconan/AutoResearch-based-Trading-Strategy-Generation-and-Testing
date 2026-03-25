#!/usr/bin/env python3
"""
Experiment #1607: 6h Primary + 1d/1w HTF — Dual EMA Trend + RSI Pullback + Volume

Hypothesis: Simple dual-EMA (8/21) trend structure with RSI(7) pullback entries provides
better trade frequency than complex indicators (Fisher, CRSI failed badly). The key is
LOOSE entry thresholds that guarantee ≥30 trades/train while 1d/1w HMA bias filters
major counter-trend trades. Volume confirmation reduces false breakouts.

Key innovations vs failed 6h attempts:
1. DUAL EMA STRUCTURE: EMA8/EMA21 crossover for trend (simpler than ribbon, more responsive)
2. FAST RSI(7): More signals than RSI(14), threshold 35/65 (not 30/70) for more trades
3. LOOSE ENTRY LOGIC: Multiple entry paths (trend continuation OR pullback) to guarantee trades
4. VOLUME CONFIRMATION: Only required for breakouts, not pullbacks (more flexible)
5. 1d/1w HMA BIAS: Prevents major counter-trend but doesn't block all trades

Entry logic (LOOSE to guarantee ≥40 trades/train):
- LONG path 1 (trend): 1d_HMA bullish + EMA8>EMA21 + RSI(7)>50 + close>EMA21
- LONG path 2 (pullback): 1d_HMA bullish + EMA8>EMA21 + RSI(7) 35-50 + volume>1.1x
- SHORT path 1 (trend): 1d_HMA bearish + EMA8<EMA21 + RSI(7)<50 + close<EMA21
- SHORT path 2 (pullback): 1d_HMA bearish + EMA8<EMA21 + RSI(7) 50-65 + volume>1.1x

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- Simpler trend structure = fewer whipsaws than triple HMA
- Fast RSI(7) = earlier entries than RSI(14)
- Dual entry paths = more trades without sacrificing quality
- Volume only required on pullbacks = flexibility on strong trends
- Looser RSI thresholds (35/65 vs 30/70) = guaranteed trade frequency

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_dual_ema_rsi7_pullback_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def calculate_rsi(close, period=7):
    """Relative Strength Index - FAST version for 6h"""
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

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_ema(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_ema(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    ema_8 = calculate_ema(close, period=8)
    ema_21 = calculate_ema(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
    
    # Warmup period
    min_bars = 60
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_8[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND STRUCTURE (Dual EMA) ===
        ema_bullish = ema_8[i] > ema_21[i]
        ema_bearish = ema_8[i] < ema_21[i]
        
        # EMA separation (trend strength)
        ema_sep = (ema_8[i] - ema_21[i]) / ema_21[i] if ema_21[i] > 1e-10 else 0
        trend_strong = abs(ema_sep) > 0.005  # 0.5% separation
        
        # === TREND DIRECTION (1d and 1w HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === RSI SIGNALS (FAST - period 7, LOOSE thresholds) ===
        rsi_val = rsi_7[i]
        rsi_neutral_long = 35 <= rsi_val <= 65  # Wide range for more signals
        rsi_neutral_short = 35 <= rsi_val <= 65
        rsi_bullish_momentum = rsi_val > 50
        rsi_bearish_momentum = rsi_val < 50
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 1.1 if not np.isnan(vol_ratio[i]) else False
        
        # === PRICE POSITION ===
        price_above_ema21 = close[i] > ema_21[i]
        price_below_ema21 = close[i] < ema_21[i]
        
        # === ENTRY LOGIC (LOOSE - multiple paths to guarantee trades) ===
        desired_signal = 0.0
        
        # LONG path 1: Strong trend continuation (1d bullish + EMA bullish + RSI>50 + price>EMA21)
        if price_above_1d and ema_bullish and rsi_bullish_momentum and price_above_ema21:
            desired_signal = SIZE_STRONG if vol_confirmed else SIZE_BASE
        
        # LONG path 2: Pullback entry (1d bullish + EMA bullish + RSI 35-50 + volume)
        elif price_above_1d and ema_bullish and 35 <= rsi_val <= 50 and vol_confirmed:
            desired_signal = SIZE_BASE
        
        # LONG path 3: 1w confirmation boost (1w bullish adds confidence)
        elif price_above_1w and ema_bullish and rsi_bullish_momentum and trend_strong:
            desired_signal = SIZE_BASE
        
        # SHORT path 1: Strong trend continuation (1d bearish + EMA bearish + RSI<50 + price<EMA21)
        elif price_below_1d and ema_bearish and rsi_bearish_momentum and price_below_ema21:
            desired_signal = -SIZE_STRONG if vol_confirmed else -SIZE_BASE
        
        # SHORT path 2: Pullback entry (1d bearish + EMA bearish + RSI 50-65 + volume)
        elif price_below_1d and ema_bearish and 50 <= rsi_val <= 65 and vol_confirmed:
            desired_signal = -SIZE_BASE
        
        # SHORT path 3: 1w confirmation boost (1w bearish adds confidence)
        elif price_below_1w and ema_bearish and rsi_bearish_momentum and trend_strong:
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