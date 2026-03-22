#!/usr/bin/env python3
"""
Experiment #446: 30m KAMA Trend + 4h HMA Filter + Volume Breakout

Hypothesis: After 431 failed experiments, the pattern is clear - complex regime 
filters and multi-signal ensembles fail on 30m due to overfitting and whipsaw.
Simple trend-following with adaptive indicators works better.

This strategy uses:
1. KAMA (Kaufman Adaptive Moving Average) - adapts to market efficiency
   - Fast KAMA(14) vs Slow KAMA(50) crossover for entries
   - KAMA reduces whipsaw in choppy markets naturally

2. 4h HMA(21) Trend Filter - via mtf_data helper
   - Long only when 30m price > 4h HMA
   - Short only when 30m price < 4h HMA
   - HMA smoother than EMA, better for HTF trend

3. Volume Confirmation - prevents false breakouts
   - Volume must be > 1.3x 20-bar average for entry
   - Confirms institutional participation

4. RSI(14) Momentum Filter - avoids extreme entries
   - Long: RSI > 45 (not oversold, momentum building)
   - Short: RSI < 55 (not overbought, momentum weakening)
   - Avoids catching falling knives

5. ATR(14) Trailing Stop at 2.0x
   - Tighter than previous 2.5x to reduce drawdown
   - Critical for 30m timeframe volatility

6. Position Sizing: 0.30 discrete
   - Conservative for 30m volatility
   - Discrete levels minimize fee churn

Why this should work on 30m:
- KAMA adapts to market conditions (less whipsaw than EMA)
- 4h HMA filter prevents counter-trend trades (proven edge)
- Volume filter reduces false breakouts
- Fewer filters = more trades (>10/year per symbol)
- Simpler than failed ensemble strategies (#434, #440, #445)

Timeframe: 30m (REQUIRED)
HTF: 4h via mtf_data helper
Position sizing: 0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_trend_4h_hma_vol_rsi_atr_v1"
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

def calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, slow_period))
    change[:slow_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(slow_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-slow_period:i+1])))
    
    volatility[:slow_period] = np.nan
    
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er[:slow_period] = np.nan
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc[:slow_period] = np.nan
    
    # Calculate KAMA
    kama[slow_period] = close[slow_period]
    for i in range(slow_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
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

def calculate_volume_ma(volume, period=20):
    """Calculate Volume Moving Average."""
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
    kama_fast = calculate_kama(close, fast_period=2, slow_period=14, smoothing_period=10)
    kama_slow = calculate_kama(close, fast_period=2, slow_period=50, smoothing_period=10)
    rsi = calculate_rsi(close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA CROSSOVER SIGNAL ===
        kama_long = kama_fast[i] > kama_slow[i]
        kama_short = kama_fast[i] < kama_slow[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # === RSI MOMENTUM FILTER ===
        rsi_long_ok = rsi[i] > 45
        rsi_short_ok = rsi[i] < 55
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Long entry: KAMA crossover + 4h bull trend + volume + RSI
        if kama_long and bull_trend_4h and vol_confirmed and rsi_long_ok:
            new_signal = SIZE
        
        # Short entry: KAMA crossover + 4h bear trend + volume + RSI
        elif kama_short and bear_trend_4h and vol_confirmed and rsi_short_ok:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
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