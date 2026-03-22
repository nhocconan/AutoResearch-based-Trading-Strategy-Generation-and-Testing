#!/usr/bin/env python3
"""
Experiment #232: 4h KAMA Trend + 1d HMA Filter + RSI Pullback + ATR Stop

Hypothesis: KAMA adapts to volatility better than EMA/HMA, reducing whipsaws in chop.
1d HMA provides stable higher-timeframe bias. RSI pullback (40-60) enters on dips in trend.
ATR trailing stop (2.5x) protects capital. This combines adaptive trend + HTF filter + timing.

Why this might work on 4h:
- KAMA adapts to market regime (fast in trends, slow in chop)
- 4h = 6 bars/day, enough trades without excessive noise
- 1d HMA prevents counter-trend trades (critical for BTC/ETH)
- RSI pullback avoids chasing breakouts (enters on retracements)
- Conservative sizing (0.30) controls drawdown in 2022-style crashes

Learning from failures:
- #226 (4h KAMA): Sharpe=-0.440 - KAMA alone, no HTF filter
- #231 (1h RSI pullback): Sharpe=-1.409 - 1h too noisy, no vol filter
- #224 (30m Supertrend): Sharpe=0.000 - 0 trades, conditions too strict
- Need: HTF filter + relaxed entry conditions + proper stoploss

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_1d_hma_rsi_pullback_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    
    er = change / (volatility + 1e-10)
    er[:er_period] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Fill initial values
    kama[:er_period] = kama[er_period]
    
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
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias (critical filter)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND STRUCTURE ===
        # KAMA slope (adaptive trend)
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # EMA structure confirmation
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 40-55 in uptrend (buy the dip)
        # Short: RSI rallied to 45-60 in downtrend (sell the rip)
        rsi_pullback_long = 35 < rsi[i] < 60
        rsi_pullback_short = 40 < rsi[i] < 65
        
        # === MOMENTUM CONFIRMATION ===
        # Price above/below 5-bar ago (simple momentum)
        momentum_long = close[i] > close[i-3]
        momentum_short = close[i] < close[i-3]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS (relaxed for trade count) ===
        # Long: 1d bullish + 4h KAMA bullish + RSI pullback + momentum
        # Only need 2 of 3 momentum filters to trigger
        long_score = 0
        if kama_bullish:
            long_score += 1
        if ema_bullish:
            long_score += 1
        if momentum_long:
            long_score += 1
        
        if bull_trend_1d and rsi_pullback_long and long_score >= 2:
            new_signal = SIZE_BASE
        
        # Short: 1d bearish + 4h KAMA bearish + RSI pullback + momentum
        short_score = 0
        if kama_bearish:
            short_score += 1
        if ema_bearish:
            short_score += 1
        if momentum_short:
            short_score += 1
        
        if bear_trend_1d and rsi_pullback_short and short_score >= 2:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction, update extremes
            elif position_side > 0 and close[i] > highest_close:
                highest_close = close[i]
            elif position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                lowest_close = close[i]
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals