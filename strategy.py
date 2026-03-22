#!/usr/bin/env python3
"""
Experiment #443: 12h Multi-Signal Ensemble with Daily Trend Filter

Hypothesis: After 442 failed experiments, the key lesson is that 12h strategies
fail when entry conditions are too strict (0 trades like #437) or too simple
(negative Sharpe). This strategy uses THREE INDEPENDENT ENTRY SIGNALS where
ANY can trigger an entry, ensuring sufficient trade frequency:

1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 1d HMA
   - Short bias when price < 1d HMA
   - More responsive than weekly, better for 12h entry timing

2. THREE INDEPENDENT ENTRY SIGNALS (any can trigger):
   a) RSI(14) MEAN REVERSION: RSI < 30 (long) or > 70 (short)
      - Must align with daily trend bias
      - Looser thresholds ensure trades in ranging markets
   
   b) KAMA(10,2,30) CROSSOVER: Fast KAMA crosses slow KAMA
      - Adaptive to volatility (Kaufman's Adaptive MA)
      - Works in both trending and ranging regimes
      - Only requires daily trend alignment
   
   c) ROC(10) MOMENTUM BURST: ROC > 4% (long) or < -4% (short)
      - Captures momentum bursts on 12h timeframe
      - Confirms entry direction with price momentum

3. NO ADX FILTER (CRITICAL CHANGE from #437):
   - ADX was blocking 60%+ of potential entries
   - Daily HMA handles regime detection instead
   - Ensures 20-50 trades/year per symbol

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crash protection

5. POSITION SIZING: 0.30 discrete (conservative for 12h volatility)
   - 30% capital per position max
   - Discrete levels minimize fee churn

Why this should work on 12h:
- Three independent signals = 3x trade opportunities vs single-signal
- No ADX filter = more entries (learned from #437 zero trades failure)
- Daily HMA = responsive enough for 12h entries, smoother than 12h MA
- KAMA adapts to volatility = fewer whipsaws in chop
- Should generate 30-60 trades/year per symbol (well above 10 minimum)
- Conservative 0.30 sizing protects against 77% BTC crash

Timeframe: 12h (REQUIRED for experiment #443)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_roc_daily_hma_ensemble_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman's Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        change = np.abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_roc(close, period=10):
    """Calculate Rate of Change (momentum)."""
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=period) * 100
    return roc.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    
    # RSI for mean reversion
    rsi = calculate_rsi(close, 14)
    
    # KAMA crossover (adaptive trend)
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=20, fast_period=5, slow_period=50)
    
    # ROC for momentum
    roc = calculate_roc(close, 10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(roc[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === SIGNAL 1: RSI MEAN REVERSION ===
        rsi_long = rsi[i] < 30  # Oversold
        rsi_short = rsi[i] > 70  # Overbought
        
        # === SIGNAL 2: KAMA CROSSOVER ===
        kama_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # === SIGNAL 3: ROC MOMENTUM BURST ===
        roc_long = roc[i] > 4.0  # Strong upward momentum
        roc_short = roc[i] < -4.0  # Strong downward momentum
        
        # === GENERATE SIGNAL (Ensemble - any signal can trigger) ===
        new_signal = 0.0
        
        # RSI MEAN REVERSION (works in any regime, must align with trend)
        if new_signal == 0.0:
            if rsi_long and bull_trend_1d:
                new_signal = SIZE
            elif rsi_short and bear_trend_1d:
                new_signal = -SIZE
        
        # KAMA CROSSOVER (adaptive trend following)
        if new_signal == 0.0:
            if kama_long and bull_trend_1d:
                new_signal = SIZE
            elif kama_short and bear_trend_1d:
                new_signal = -SIZE
        
        # ROC MOMENTUM (confirms strong moves)
        if new_signal == 0.0:
            if roc_long and bull_trend_1d:
                new_signal = SIZE
            elif roc_short and bear_trend_1d:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
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