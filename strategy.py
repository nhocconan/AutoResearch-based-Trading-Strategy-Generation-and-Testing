#!/usr/bin/env python3
"""
Experiment #456: 12h Primary + 1d HTF — KAMA Adaptive Trend + Choppiness Regime + RSI Entries

Hypothesis: Building on #452 (Sharpe=0.209 with KAMA), improve with cleaner regime logic.
Key features:
1. KAMA (Kaufman Adaptive MA) - adapts to volatility, less whipsaw than HMA/EMA
2. Choppiness Index for regime detection (CHOP>61.8=range, <38.2=trend)
3. RSI(14) for entry timing with relaxed thresholds (35/65)
4. 1d HMA for HTF trend bias
5. ATR(14) trailing stop at 2.5x
6. Position size: 0.25 base, 0.30 on strong confluence, discrete levels

Target: Sharpe > 0.612, 80-200 trades over 4-year train, DD < -35%
Timeframe: 12h (higher TF = fewer trades, less fee drag, proven in history)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_rsi_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.nansum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_di_s / (atr + 1e-10)
        minus_di = 100.0 * minus_di_s / (atr + 1e-10)
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    kama_fast = calculate_kama(close, period=10, fast=2, slow=30)
    kama_slow = calculate_kama(close, period=20, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = 1.0
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% base position size for 12h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(adx[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 55.0  # Range market (relaxed from 61.8)
        regime_trend = chop[i] < 45.0  # Trending market (relaxed from 38.2)
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (KAMA crossover) ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === RSI SIGNALS (Relaxed thresholds for more trades) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_oversold = rsi[i] < 30.0
        rsi_extreme_overbought = rsi[i] > 70.0
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 20.0  # Relaxed from 25
        adx_weak = adx[i] < 25.0
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 55) — MEAN REVERSION ===
        if regime_chop:
            # Long: RSI oversold + price above 1d HMA (bullish bias)
            if rsi_oversold and price_above_hma_1d:
                desired_signal = position_size * 0.8
                if rsi_extreme_oversold:
                    desired_signal = position_size
            # Short: RSI overbought + price below 1d HMA (bearish bias)
            elif rsi_overbought and price_below_hma_1d:
                desired_signal = -position_size * 0.8
                if rsi_extreme_overbought:
                    desired_signal = -position_size
        
        # === REGIME 2: TRENDING (CHOP < 45) — TREND FOLLOW ===
        elif regime_trend:
            # Long: KAMA bullish + RSI not overbought + ADX strong
            if kama_bullish and not rsi_overbought:
                if adx_strong:
                    desired_signal = position_size
                else:
                    desired_signal = position_size * 0.7
            # Short: KAMA bearish + RSI not oversold + ADX strong
            elif kama_bearish and not rsi_oversold:
                if adx_strong:
                    desired_signal = -position_size
                else:
                    desired_signal = -position_size * 0.7
        
        # === REGIME 3: TRANSITION (45-55) — HYBRID ===
        else:
            # KAMA crossover with RSI confirmation
            if kama_bullish and rsi_oversold:
                desired_signal = position_size * 0.6
            elif kama_bearish and rsi_overbought:
                desired_signal = -position_size * 0.6
            elif kama_bullish and price_above_hma_1d:
                desired_signal = position_size * 0.5
            elif kama_bearish and price_below_hma_1d:
                desired_signal = -position_size * 0.5
        
        # === CAP SIGNAL TO MAX 0.35 ===
        if desired_signal > 0.35:
            desired_signal = 0.35
        elif desired_signal < -0.35:
            desired_signal = -0.35
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === RSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and rsi[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (kama_bullish or price_above_hma_1d):
                desired_signal = position_size * 0.5
            elif position_side < 0 and (kama_bearish or price_below_hma_1d):
                desired_signal = -position_size * 0.5
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.18:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.18:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.15
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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