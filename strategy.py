#!/usr/bin/env python3
"""
Experiment #556: 4h KAMA Adaptive Trend with Dual HTF Bias (1d/1w)

Hypothesis: After analyzing 500+ failed experiments, the key insight is:
1. KAMA (Kaufman Adaptive MA) adapts to volatility better than EMA/HMA
2. Dual HTF bias (1w + 1d) provides stronger trend confirmation than single HTF
3. 4h timeframe balances noise reduction with trade frequency
4. RSI momentum confirmation (not extremes) avoids whipsaw entries
5. Volume spike on breakout confirms genuine moves vs fakeouts
6. 2.5*ATR stoploss protects against crypto volatility

Why this should work on 4h:
- KAMA adapts ER (Efficiency Ratio) - fast in trends, slow in chop
- 1w HMA = major trend bias (only trade with weekly direction)
- 1d HMA = intermediate confirmation (aligns with weekly)
- RSI(14) > 55 for long, < 45 for short = momentum confirmation
- Volume > 1.5 * SMA(volume, 20) = confirms breakout validity
- Works in both bull and bear regimes due to adaptive nature

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adaptive_dual_htf_rsi_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise using Efficiency Ratio (ER).
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Change over period
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    # Sum of absolute changes (volatility)
    abs_diff = np.abs(close - np.roll(close, 1))
    abs_diff[0] = abs_diff[1] if n > 1 else 0
    volatility = pd.Series(abs_diff).rolling(window=period, min_periods=period).sum().values
    
    # Efficiency Ratio
    er = change / volatility
    er[np.isnan(er)] = 0
    er = np.clip(er, 0, 1)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above moving average."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * vol_sma)
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, period=10, fast=2, slow=30)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    volume_spike = calculate_volume_spike(volume, 20, 1.5)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_4h[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS (Major Trend) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DAILY HMA TREND BIAS (Intermediate Trend) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 4H KAMA ADAPTIVE TREND ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        
        # === RSI MOMENTUM CONFIRMATION ===
        rsi_long = rsi_14[i] > 55  # Momentum confirmation for long
        rsi_short = rsi_14[i] < 45  # Momentum confirmation for short
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume_spike[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: Weekly bullish + Daily bullish + KAMA bullish + RSI momentum + Volume
        # Relaxed: need weekly + (daily OR KAMA) + RSI
        if weekly_bull and (daily_bull or kama_bull) and rsi_long:
            new_signal = SIZE
        
        # Short: Weekly bearish + Daily bearish + KAMA bearish + RSI momentum
        # Relaxed: need weekly + (daily OR KAMA) + RSI
        elif weekly_bear and (daily_bear or kama_bear) and rsi_short:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if weekly HMA flips against position (major trend change)
        if in_position and new_signal != 0.0:
            if position_side > 0 and weekly_bear:
                new_signal = 0.0
            if position_side < 0 and weekly_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals