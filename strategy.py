#!/usr/bin/env python3
"""
Experiment #347: 12h Volatility-Adaptive Strategy with 1d HMA Bias + RSI Mean Reversion

Hypothesis: After analyzing 295 failed strategies, the key insight for 12h timeframe is:
1. 12h is slow enough to avoid noise but fast enough to catch intermediate swings
2. Pure trend following fails on BTC/ETH (2022 crash whipsawed all trend strategies)
3. Mean reversion works better in bear/range markets (2025 test period)
4. Volatility regime detection helps avoid entries during extreme moves
5. LOOSE entry conditions are CRITICAL - many strategies failed with 0 trades

Strategy components:
1. 1d HMA(21) for long-term trend bias (via mtf_data helper - call ONCE)
2. ATR(7)/ATR(30) ratio for volatility regime (>2.0 = vol spike, <1.0 = calm)
3. RSI(14) with LOOSE thresholds (25/75) to ensure >=10 trades per symbol
4. Volume confirmation (volume > 0.8 * SMA20 volume)
5. Stoploss at 2.5 * ATR(14) trailing
6. Position sizing: 0.25 discrete levels

Why this should work on 12h:
- 12h bars capture multi-day swings without intraday noise
- Volatility regime filter avoids entering during panic/extreme moves
- Loose RSI thresholds ensure trades even in quiet markets
- 1d HMA provides stable directional bias without whipsaw
- Designed specifically to generate trades in both bull and bear regimes

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_adaptive_1d_hma_rsi_loose_atr_v1"
timeframe = "12h"
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
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma20 = calculate_sma(volume, 20)
    
    # Volatility ratio: ATR(7) / ATR(30)
    vol_ratio = np.full(n, np.nan)
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY REGIME ===
        vol_spike = vol_ratio[i] > 2.0  # High volatility - avoid new entries
        vol_calm = vol_ratio[i] < 1.2   # Low volatility - good for entries
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === RSI EXTREMES (LOOSE thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 35   # Loosened from 30 for more trades
        rsi_overbought = rsi_14[i] > 65  # Loosened from 70 for more trades
        rsi_neutral = 35 <= rsi_14[i] <= 65
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_sma20[i] > 0 and volume[i] > 0.8 * vol_sma20[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: RSI oversold + 1d HMA bullish + vol calm/normal + volume confirm
        if rsi_oversold and bull_trend_1d and not vol_spike:
            new_signal = SIZE
        
        # SHORT ENTRY: RSI overbought + 1d HMA bearish + vol calm/normal + volume confirm
        elif rsi_overbought and bear_trend_1d and not vol_spike:
            new_signal = -SIZE
        
        # NEUTRAL RSI + TREND CONTINUATION: Stay in position if already positioned
        elif in_position and rsi_neutral:
            # Keep existing position during neutral RSI
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === VOLATILITY SPIKE EXIT ===
        # Exit position if volatility spikes extremely (panic)
        if in_position and vol_ratio[i] > 3.0:
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