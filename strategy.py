#!/usr/bin/env python3
"""
Experiment #225: 1h KAMA Crossover + 4h HMA Trend + ADX Filter + ATR Stop

Hypothesis: 1h timeframe with adaptive KAMA entries filtered by 4h HMA trend
will capture medium-term swings while avoiding 2022-style crash drawdowns.
KAMA adapts to volatility (better than EMA in ranges), 4h HMA provides stable
trend bias, ADX > 15 ensures sufficient trade count, and ATR stop protects
against reversals.

Why this might work better than failed experiments:
- #219 (1h Fisher): Sharpe=0.000 - 0 trades, conditions too strict
- #224 (30m Supertrend): Sharpe=0.000 - 0 trades, Supertrend whipsaws
- KAMA adapts to market regime (ER-based smoothing)
- Lower ADX threshold (15 vs 25) ensures trade generation
- 4h HMA filter proven in best strategy (Sharpe=0.478)

Key design choices:
- Timeframe: 1h (REQUIRED for this experiment)
- HTF: 4h HMA via mtf_data helper (call ONCE before loop)
- Entry: KAMA(10) > KAMA(30) for long, opposite for short
- Filter: ADX > 15 (trending), price vs 4h HMA (trend alignment)
- Stoploss: 2.5 * ATR(14) trailing
- Position sizing: 0.30 discrete (controls drawdown in crashes)

Learning from failures:
- Must generate trades! Loosen filters if needed
- ADX > 15 not > 25 (ensures 10+ trades)
- No RSI extreme filters (RSI 40-60 range, not 30/70)
- KAMA better than EMA for adaptive trend following
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_adx_atr_v1"
timeframe = "1h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    High ER = trending (use fast SC), Low ER = noisy (use slow SC)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er_num = np.zeros(n)
    er_den = np.zeros(n)
    
    for i in range(er_period, n):
        er_num[i] = np.abs(close[i] - close[i - er_period])
        er_den[i] = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
    
    er_den = np.where(er_den == 0, 1e-10, er_den)
    er = er_num / er_den
    er[:er_period] = er[er_period] if er_period < n else 0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[0] = close[0]
    
    # Calculate KAMA iteratively
    for i in range(1, n):
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
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # KAMA for adaptive trend following (fast and slow)
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_slow = calculate_kama(close, er_period=10, fast_period=5, slow_period=30)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 15 = trending market (lower threshold to ensure trades)
        trend_strength = adx[i] > 15
        
        # === KAMA CROSSOVER ===
        # Fast KAMA > Slow KAMA = bullish momentum
        # Fast KAMA < Slow KAMA = bearish momentum
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === RSI MOMENTUM FILTER ===
        # RSI > 45 = bullish momentum (not extreme)
        # RSI < 55 = bearish momentum (not extreme)
        rsi_bullish = rsi[i] > 45
        rsi_bearish = rsi[i] < 55
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + ADX trending + KAMA bullish + RSI confirmation
        # Using flexible conditions to ensure enough trades
        if bull_trend_4h and trend_strength and kama_bullish and rsi_bullish:
            new_signal = SIZE_BASE
        
        # Short: 4h bearish + ADX trending + KAMA bearish + RSI confirmation
        if bear_trend_4h and trend_strength and kama_bearish and rsi_bearish:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and new_signal != 0.0:
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
        
        # === EXIT ON SIGNAL REVERSAL ===
        # If we have a position and signal flips, exit first
        if in_position and new_signal != 0.0:
            if np.sign(new_signal) != position_side:
                # Position reversal - exit current first
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
                # Will re-enter on next bar with new signal
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        
        signals[i] = new_signal
    
    return signals