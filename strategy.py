#!/usr/bin/env python3
"""
Experiment #201: 1h KAMA Trend + 4h HMA Filter + ADX Regime + ATR Stop

Hypothesis: 1h KAMA adapts to volatility (fast in trends, slow in ranges),
4h HMA provides stable higher-timeframe bias, ADX > 20 filters choppy markets,
and ATR trailing stop protects against reversals. This combines the proven
4h KAMA approach (current best Sharpe=0.478) but on faster 1h timeframe
for more trade opportunities while maintaining HTF trend alignment.

Why 1h might work better than recent failures:
- 1h has 4x more bars than 4h = more trade opportunities
- KAMA adapts to regime automatically (no complex regime detection needed)
- 4h HMA filter prevents counter-trend trades (learned from #193, #199 failures)
- ADX > 20 (not 25+) ensures sufficient trade count
- Conservative sizing (0.28) controls drawdown

Learning from failures:
- #193 (15m pullback): Sharpe=-3.489 - too fast, too many whipsaws
- #195 (1h regime): Sharpe=-1.100 - over-complicated with too many filters
- #199 (15m KAMA): Sharpe=-4.763 - 15m too noisy even with KAMA
- #189 (1h regime): Sharpe=-0.064 - regime detection overfit
- Simple is better: KAMA + HTF filter + ADX + ATR stop

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
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
    
    # Calculate True Range and Directional Movement
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
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility:
    - Fast EMA (span=2) when trending (high Efficiency Ratio)
    - Slow EMA (span=30) when ranging (low Efficiency Ratio)
    
    This is proven to work well in crypto (current best strategy uses KAMA).
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    # ER = |change| / sum(|individual changes|) over period
    # ER ranges from 0 (noisy/ranging) to 1 (strong trend)
    er = np.zeros(n)
    for i in range(period, n):
        change = np.abs(close[i] - close[i-period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = change / noise
        else:
            er[i] = 0
    
    # Fill initial ER values
    er[:period] = er[period] if period < n else 0
    
    # Calculate smoothing constant (SC)
    # SC = ER * (fast_sc - slow_sc) + slow_sc
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    # KAMA[i] = KAMA[i-1] + SC[i] * (close[i] - KAMA[i-1])
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    # Use 0.28 to balance trade frequency with drawdown control
    SIZE_BASE = 0.28
    
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
        
        if np.isnan(adx[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias (proven in best strategy)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 20 = trending market (lower than 25 to ensure trades on 1h)
        trend_strength = adx[i] > 20
        
        # === KAMA TREND SIGNAL ===
        # KAMA adapts to volatility - fast in trends, slow in ranges
        # Price above KAMA = bullish, below = bearish
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # === EMA CONFIRMATION ===
        # EMA21 > EMA50 = bullish trend structure
        # EMA21 < EMA50 = bearish trend structure
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI MOMENTUM ===
        # RSI > 45 = not oversold (avoid catching falling knives)
        # RSI < 55 = not overbought (avoid chasing tops)
        rsi_not_oversold = rsi[i] > 45
        rsi_not_overbought = rsi[i] < 55
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + ADX trending + KAMA bullish + (EMA bullish OR RSI ok)
        # Using flexible conditions to ensure enough trades (learned from failures)
        if bull_trend_4h and trend_strength and kama_bullish:
            # Need at least one momentum confirmation (flexible OR)
            if ema_bullish or rsi_not_oversold:
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + ADX trending + KAMA bearish + (EMA bearish OR RSI ok)
        if bear_trend_4h and trend_strength and kama_bearish:
            # Need at least one momentum confirmation (flexible OR)
            if ema_bearish or rsi_not_overbought:
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
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals