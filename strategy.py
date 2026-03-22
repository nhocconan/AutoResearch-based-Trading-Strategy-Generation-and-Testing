#!/usr/bin/env python3
"""
Experiment #440: 30m Supertrend + RSI Pullback with 4h HMA Trend Filter

Hypothesis: After 439 experiments, the key insight is that 30m timeframe offers
the sweet spot between noise filtering (vs 5m/15m) and trade frequency (vs 4h/1d).
This strategy combines:

1. SUPERTREND(10, 3.0) on 30m: Clear trend direction with ATR-based stops
   - Proven to work better than EMA crossovers in crypto
   - ATR multiplier of 3.0 avoids whipsaws while catching moves

2. 4h HMA(21) TREND BIAS via mtf_data helper:
   - Long only when 30m price > 4h HMA (aligns with HTF trend)
   - Short only when 30m price < 4h HMA
   - HMA smoother than EMA, critical for HTF trend detection

3. RSI(14) PULLBACK ENTRIES:
   - Long: Supertrend bullish + RSI < 45 (pullback in uptrend)
   - Short: Supertrend bearish + RSI > 55 (pullback in downtrend)
   - Looser than 30/70 to ensure sufficient trades on 30m

4. VOLUME CONFIRMATION:
   - Volume > SMA(volume, 20) * 1.1 on entry bar
   - Filters false breakouts and low-liquidity moves

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for crash protection (2022-style moves)

6. POSITION SIZING: 0.28 discrete (conservative for 30m volatility)
   - Max 28% capital per position
   - Discrete levels minimize fee churn

Why this should work on 30m:
- Supertrend provides clear trend direction (better than EMA)
- 4h HMA filter prevents counter-trend disasters
- RSI pullback ensures entries on retracements (not chasing)
- Volume filter reduces false signals
- Should generate 20-50 trades/year per symbol (sufficient frequency)
- Conservative sizing protects against 2022-style crashes

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_rsi_4h_hma_vol_atr_v1"
timeframe = "30m"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_line, trend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            if trend[i-1] == 1:
                if close[i] < lower_band[i]:
                    trend[i] = -1
                    supertrend[i] = upper_band[i]
                else:
                    trend[i] = 1
                    supertrend[i] = max(lower_band[i], supertrend[i-1])
            else:
                if close[i] > upper_band[i]:
                    trend[i] = 1
                    supertrend[i] = lower_band[i]
                else:
                    trend[i] = -1
                    supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    return supertrend, trend

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

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend_line, supertrend_trend = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
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
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend_trend[i]) or np.isnan(supertrend_line[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        supertrend_bullish = supertrend_trend[i] == 1
        supertrend_bearish = supertrend_trend[i] == -1
        
        # === RSI PULLBACK ===
        rsi_pullback_long = rsi[i] < 45  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 55  # Pullback in downtrend
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma[i] * 1.1 if vol_sma[i] > 0 else False
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: Supertrend bullish + 4h HMA bull + RSI pullback + volume
        if supertrend_bullish and bull_trend_4h and rsi_pullback_long and volume_confirmed:
            new_signal = SIZE
        
        # SHORT: Supertrend bearish + 4h HMA bear + RSI pullback + volume
        elif supertrend_bearish and bear_trend_4h and rsi_pullback_short and volume_confirmed:
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
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === SUPERTREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and supertrend_bearish:
                new_signal = 0.0
            if position_side < 0 and supertrend_bullish:
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