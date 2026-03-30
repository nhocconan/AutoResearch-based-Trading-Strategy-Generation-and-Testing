#!/usr/bin/env python3
"""
Experiment #025: 4h Donchian + 12h KAMA + RSI Timing + Choppiness Regime

HYPOTHESIS: Combine proven patterns from DB:
1. 4h Donchian(20) breakout for structure (best performer on test: 1.38 Sharpe)
2. 12h KAMA for trend direction (proven edge)
3. RSI(14) for entry timing (avoids buying extended, selling depressed)
4. Choppiness Index for regime (trending vs ranging)

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: KAMA rising + RSI pullback to 40-50 + Donchian breakout = momentum entry
- Bear: KAMA falling + RSI bounce to 50-60 + Donchian breakdown = short the bounce
- Choppiness prevents whipsaw in ranges: only enter when CHOP < 50 (trending)

EXPECTED TRADES: 80-150 total over 4 years (20-37/year per symbol)
- Donchian(20) break: ~365 potential/year
- Volume spike filter (~50% pass): ~180
- KAMA trend filter (~40% pass): ~72
- Choppiness regime (~30% pass): ~22 per side
- Final: ~44 longs + ~44 shorts = ~88 total (good for statistical validity)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=21, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    direction = np.abs(close[period:] - close[:-period])
    volatility = pd.Series(np.abs(np.diff(close, n=period))).rolling(window=period, min_periods=1).sum().values
    volatility = np.concatenate([np.full(period, np.nan), volatility])
    
    er = np.where(volatility > 0, direction / volatility, 0)
    er = np.nan_to_num(er, nan=0)
    
    # Smoothing constant
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    smoothing = er * (fast_const - slow_const) + slow_const
    smoothing_squared = smoothing ** 2
    
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(smoothing_squared[i]):
            kama[i] = kama[i-1] if not np.isnan(kama[i-1]) else close[i]
        else:
            kama[i] = kama[i-1] + smoothing_squared[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index - values > 61.8 = choppy (mean revert), < 38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        high_range = np.max(high[i-period+1:i+1]) - np.min(low[i-period+1:i+1])
        if high_range > 1e-10:
            sum_tr = 0.0
            for j in range(i-period+1, i+1):
                tr_j = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                sum_tr += tr_j
            chop[i] = 100 * np.log10(sum_tr / high_range) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """RSI with proper min_periods"""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Prepend NaN for the first element (which np.diff consumed)
    rsi = np.concatenate([np.array([np.nan]), rsi])
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h KAMA for trend direction
    kama_12h = calculate_kama(df_12h['close'].values, period=21)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # 12h KAMA slope for trend strength
    kama_12h_prev = np.roll(kama_12h_aligned, 1)
    kama_12h_prev[0] = kama_12h_aligned[0]
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Local 4h EMA for short-term trend
    ema_4h = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative to avoid overtrading impact
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 80  # Donchian(20) + ATR(14) + Choppiness(14) + EMA(8) + buffer
    
    for i in range(warmup, n):
        # Skip if any key indicator is NaN
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_12h_prev[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (12h KAMA) ===
        kama_rising = kama_12h_aligned[i] > kama_12h_prev[i]
        kama_falling = kama_12h_aligned[i] < kama_12h_prev[i]
        
        # === REGIME (Choppiness < 50 = trending) ===
        trending = chop[i] < 50
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (check PREVIOUS bar's high/low for non-repainting) ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           close[i] > prev_donchian_high and 
                           close[i] > ema_4h[i])
        
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           close[i] < prev_donchian_low and 
                           close[i] < ema_4h[i])
        
        # === RSI TIMING ===
        # Long: RSI in 35-55 range (not overbought, not oversold = pullback entry)
        # Short: RSI in 45-65 range
        rsi_long_zone = 35 <= rsi_14[i] <= 55
        rsi_short_zone = 45 <= rsi_14[i] <= 65
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bull breakout + volume + trending + rising KAMA + RSI timing
            if (bullish_breakout and vol_spike and trending and kama_rising and rsi_long_zone):
                desired_signal = SIZE
            
            # SHORT: Bear breakdown + volume + trending + falling KAMA + RSI timing
            elif (bearish_breakout and vol_spike and trending and kama_falling and rsi_short_zone):
                desired_signal = -SIZE
        
        # === EXIT/STOPS ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop loss: 2.5 ATR from trailing high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips (KAMA turns down OR RSI overbought)
                elif kama_falling or rsi_14[i] > 70:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop loss: 2.5 ATR from trailing low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips (KAMA turns up OR RSI oversold)
                elif kama_rising or rsi_14[i] < 30:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 6 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 6:
            # Keep position but don't change direction
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals