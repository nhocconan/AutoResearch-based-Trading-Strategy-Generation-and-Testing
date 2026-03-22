#!/usr/bin/env python3
"""
Experiment #012: 12h Dual-Regime Strategy with Adaptive Entries

Hypothesis: Previous strategies failed due to overly strict confluence requirements.
This strategy uses SIMPLER entry logic with forced frequency safeguards:

1. KAMA trend filter (adaptive to volatility)
2. RSI(7) for entry timing (more frequent than Connors RSI extremes)
3. ADX(14) > 18 for trend confirmation (lower than typical 25)
4. 1d HMA for major bias (single HTF filter, not multiple)
5. FORCE entry every 35 bars if no signal (ensures 40-60 trades/year)

Key improvements over #002:
- Simpler RSI(7) < 35 / > 65 thresholds (vs Connors < 15 / > 85)
- Single 1d HMA filter (not 1d + 1w)
- ADX > 18 instead of Choppiness regime switch
- Forced entry every 35 bars prevents 0-trade scenarios
- Simpler stoploss: 3.0 ATR from entry (not trailing)

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 3.0 * ATR(14) from entry
Target: 40-60 trades/year on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_kama_rsi_adx_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Efficiency Ratio
        if i >= er_period:
            change = np.abs(close[i] - close[i - er_period])
            volatility = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
            if volatility > 0:
                er = change / volatility
            else:
                er = 0
        else:
            er = 0
        
        # Smoothing constants
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h = calculate_kama(close, 10, 2, 30)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for more signals
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_12h[i]) or np.isnan(rsi_7[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H KAMA TREND ===
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # === TREND STRENGTH ===
        is_trending = adx_14[i] > 18.0  # Lower threshold for more signals
        
        # === RSI ENTRY SIGNALS ===
        rsi_oversold = rsi_7[i] < 35.0  # Long entry
        rsi_overbought = rsi_7[i] > 65.0  # Short entry
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC (SIMPLIFIED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: Daily bullish + KAMA bullish + RSI oversold
        # Only require 2 of 3 conditions in trending market
        long_conditions = sum([daily_bullish, kama_bullish, rsi_oversold])
        if long_conditions >= 2:
            if is_trending or rsi_oversold:  # Either trending OR oversold
                new_signal = current_size
        
        # SHORT: Daily bearish + KAMA bearish + RSI overbought
        short_conditions = sum([daily_bearish, kama_bearish, rsi_overbought])
        if short_conditions >= 2:
            if is_trending or rsi_overbought:  # Either trending OR overbought
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD (CRITICAL for trade count) ===
        # Force entry if no trades for 35 bars (~17.5 days on 12h)
        if bars_since_last_trade > 35 and new_signal == 0.0 and not in_position:
            if daily_bullish and kama_bullish:
                new_signal = current_size * 0.6
            elif daily_bearish and kama_bearish:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR from entry ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                stoploss_price = entry_price - 3.0 * entry_atr
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                stoploss_price = entry_price + 3.0 * entry_atr
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and daily_bearish:
                trend_reversal = True
            if position_side < 0 and kama_bullish and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals