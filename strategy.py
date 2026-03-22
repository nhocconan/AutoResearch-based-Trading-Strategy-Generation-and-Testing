#!/usr/bin/env python3
"""
Experiment #479: 12h Volatility Breakout with Daily HMA Trend Filter

Hypothesis: After analyzing 478 failed experiments, the pattern is clear:
1. Complex regime filters (CHOP, multiple conditions) reduce trade count too much
2. 12h timeframe needs LOOSER entry thresholds to generate sufficient trades
3. Volatility contraction + breakout works better than pure mean-reversion on 12h
4. Daily HMA provides cleaner trend bias than weekly (less lag, more signals)

Strategy Design:
1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Bull: price > 1d HMA (favor long breakouts)
   - Bear: price < 1d HMA (favor short breakouts)

2. BOLLINGER BANDWIDTH CONTRACTION:
   - BB Width < 20th percentile of last 100 bars = squeeze
   - Squeeze precedes volatility expansion (breakout)

3. VOLATILITY BREAKOUT ENTRY:
   - Long: Bull regime + BB squeeze + price breaks BB upper + ADX > 15
   - Short: Bear regime + BB squeeze + price breaks BB lower + ADX > 15

4. RSI(7) CONFIRMATION (faster than RSI14):
   - Long: RSI(7) > 45 (momentum confirming)
   - Short: RSI(7) < 55 (momentum confirming)

5. ATR(14) TRAILING STOP at 2.5x:
   - Tighter stop for 12h timeframe
   - Signal → 0 when price moves 2.5*ATR against position

6. POSITION SIZING: 0.25 discrete
   - Conservative for 12h volatility
   - Discrete levels minimize fee churn

Why this should work on 12h:
- BB squeeze + breakout captures volatility expansion (proven pattern)
- Daily HMA trend filter avoids counter-trend trades
- RSI(7) faster than RSI(14) = more signals on 12h
- ADX > 15 (not 25) = looser filter = more trades
- Should generate 30-50 trades/year per symbol

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_squeeze_breakout_daily_hma_rsi7_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_rsi(close, period=7):
    """Calculate Relative Strength Index (faster period for 12h)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    bandwidth = (upper - lower) / sma
    
    return upper.values, lower.values, bandwidth.values

def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate percentile rank of BB bandwidth over lookback period."""
    n = len(bandwidth)
    bb_pct = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = bandwidth[i-lookback:i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            rank = np.sum(valid_window <= bandwidth[i]) / len(valid_window)
            bb_pct[i] = rank * 100
    
    return bb_pct

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
    rsi = calculate_rsi(close, 7)  # Faster RSI for 12h
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_pct = calculate_bb_percentile(bb_bandwidth, 100)
    
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
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_pct[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # === BOLLINGER BANDWIDTH SQUEEZE ===
        bb_squeeze = bb_pct[i] < 20  # Bottom 20% = squeeze
        
        # === VOLATILITY BREAKOUT ENTRY ===
        new_signal = 0.0
        
        # LONG: Bull regime + BB squeeze + breakout above upper band + momentum
        if bull_regime and bb_squeeze:
            if close[i] > bb_upper[i-1] if i > 0 else bb_upper[i]:
                if adx[i] > 15 and rsi[i] > 45:
                    new_signal = SIZE
        
        # SHORT: Bear regime + BB squeeze + breakout below lower band + momentum
        if bear_regime and bb_squeeze:
            if close[i] < bb_lower[i-1] if i > 0 else bb_lower[i]:
                if adx[i] > 15 and rsi[i] < 55:
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if daily trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
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