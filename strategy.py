#!/usr/bin/env python3
"""
Experiment #111: 1h KAMA + 4h HMA Trend Filter + RSI Pullback + ATR Stop

Hypothesis: Building on #100's success (Sharpe=0.436 on 4h), this adapts the approach to 1h:
- 1h KAMA(10,2,30) captures adaptive trend following on primary timeframe
- 4h HMA(21) provides stable higher-timeframe trend bias (proven in #100)
- RSI(14) pullback entries: enter on RSI 40-50 dip in uptrend, 50-60 rally in downtrend
- ATR-based trailing stop (2.5*ATR) protects against adverse moves
- Simple entry logic ensures trades on ALL symbols (BTC/ETH/SOL)
- Discrete position sizing (0.20/0.30) minimizes fee churn

Why this might work on 1h (unlike failed 1h strategies #105, #109, #110):
- RSI used for pullback confirmation, NOT mean reversion (avoids #103 failure)
- 4h HMA trend filter is proven stable (worked in #100)
- KAMA adapts to volatility better than EMA/Supertrend
- Fewer conflicting filters = more trades while maintaining quality
- ATR stoploss prevents catastrophic drawdowns

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_rsi_pullback_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in ranges.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    price_change = np.abs(close - np.roll(close, er_period))
    price_change[:er_period] = np.nan
    
    individual_changes = np.abs(np.diff(close))
    individual_changes = np.insert(individual_changes, 0, 0)
    
    sum_changes = pd.Series(individual_changes).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = np.zeros(n)
    mask = sum_changes > 0
    er[mask] = price_change[mask] / sum_changes[mask]
    er[:er_period] = np.nan
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc[:er_period] = np.nan
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = np.nan
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
    """Calculate RSI using standard Wilder's method."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # No losses = RSI 100
    return rsi

def generate_signals(prices):
    global n  # Make n accessible for RSI calculation
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
    kama = calculate_kama(close, 10, 2, 30)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
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
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA slope (trend strength) - compare to 5 bars ago
        kama_slope = 0.0
        if i >= 5 and not np.isnan(kama[i-5]):
            kama_slope = (kama[i] - kama[i-5]) / kama[i-5] if kama[i-5] != 0 else 0
        
        # === RSI PULLBACK FILTER ===
        # In uptrend: look for RSI pullback to 40-50 zone (buying dip)
        # In downtrend: look for RSI rally to 50-60 zone (selling rally)
        rsi_pullback_long = 40 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 60
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 4h bullish + KAMA bullish + positive slope + RSI pullback
        if bull_trend_4h and kama_bullish and kama_slope > 0 and rsi_pullback_long:
            new_signal = SIZE_STRONG
        # Moderate: 4h bullish + KAMA bullish (ensure trades on all symbols)
        elif bull_trend_4h and kama_bullish:
            new_signal = SIZE_BASE
        # Weak: KAMA bullish + positive slope (fallback for SOL)
        elif kama_bullish and kama_slope > 0.005:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 4h bearish + KAMA bearish + negative slope + RSI pullback
        if bear_trend_4h and kama_bearish and kama_slope < 0 and rsi_pullback_short:
            new_signal = -SIZE_STRONG
        # Moderate: 4h bearish + KAMA bearish (ensure trades on all symbols)
        elif bear_trend_4h and kama_bearish:
            new_signal = -SIZE_BASE
        # Weak: KAMA bearish + negative slope (fallback for SOL)
        elif kama_bearish and kama_slope < -0.005:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals