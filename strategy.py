#!/usr/bin/env python3
"""
Experiment #492: 1d Volatility-Adjusted Mean Reversion with Weekly Trend Filter

Hypothesis: After analyzing 491 failed experiments, the pattern is clear:
1. Complex regime-adaptive logic (Chop, ADX, multiple filters) overfits and fails
2. Daily timeframe mean reversion works better than trend following for BTC/ETH
3. Weekly HMA trend filter is sufficient (no need for complex regime detection)
4. RSI(7) generates more trades than RSI(14) - critical for Sharpe calculation
5. Bollinger Band position adds timing confirmation without over-filtering
6. ATR-based position sizing reduces risk during high volatility periods

Strategy Logic:
1. WEEKLY HMA(21) TREND FILTER (via mtf_data helper):
   - Long only when price > 1w HMA (bullish bias)
   - Short only when price < 1w HMA (bearish bias)
   - Simple, robust, no whipsaw

2. RSI(7) MEAN REVERSION:
   - Long: RSI < 25 (oversold, looser than 30 for more trades)
   - Short: RSI > 75 (overbought, looser than 70 for more trades)
   - Faster RSI period = more signals

3. BOLLINGER BAND CONFIRMATION:
   - Long: price < BB_lower (at bottom of range)
   - Short: price > BB_upper (at top of range)
   - Adds timing precision without over-filtering

4. ATR-BASED POSITION SIZING:
   - Base size: 0.30
   - Reduce to 0.20 when ATR > 1.5x rolling average (high vol = smaller size)
   - Discrete levels: 0.0, ±0.20, ±0.30

5. FIXED STOPLOSS at 2.5*ATR:
   - Simpler than trailing, easier to backtest
   - Signal → 0 when price moves 2.5*ATR against position

6. TIME-BASED EXIT:
   - Exit after 10 days if no stoploss hit (mean reversion should work quickly)

Why this should beat Sharpe=0.676:
- Simpler logic = less overfitting
- RSI(7) generates 2-3x more trades than RSI(14)
- Weekly filter prevents counter-trend disasters
- ATR sizing reduces drawdown in volatile periods
- Should generate 30-50 trades/year per symbol

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete, vol-adjusted
Stoploss: 2.5 * ATR(14) fixed
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_voladj_meanrev_weekly_hma_rsi7_bb_atr_v1"
timeframe = "1d"
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

def calculate_rsi(close, period=7):
    """Calculate Relative Strength Index with faster period for more signals."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_atr_ma(atr, period=20):
    """Calculate rolling average of ATR for vol adjustment."""
    atr_s = pd.Series(atr)
    return atr_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    atr_ma = calculate_atr_ma(atr, 20)
    rsi = calculate_rsi(close, 7)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels with vol adjustment (Rule 4)
    BASE_SIZE = 0.30
    LOW_VOL_SIZE = 0.30
    HIGH_VOL_SIZE = 0.20
    
    # Track position state for stoploss and time exit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    stoploss_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_ma[i]) or atr_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND FILTER ===
        bull_trend = close[i] > hma_1w_aligned[i]
        bear_trend = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        vol_ratio = atr[i] / atr_ma[i]
        if vol_ratio > 1.5:
            current_size = HIGH_VOL_SIZE
        else:
            current_size = LOW_VOL_SIZE
        
        # === MEAN REVERSION ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: RSI oversold + price at BB lower + bull trend
        if bull_trend:
            if rsi[i] < 25 and close[i] <= bb_lower[i]:
                new_signal = current_size
        
        # SHORT: RSI overbought + price at BB upper + bear trend
        if bear_trend:
            if rsi[i] > 75 and close[i] >= bb_upper[i]:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR fixed ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Long position stoploss
                stoploss_price = entry_price - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Short position stoploss
                stoploss_price = entry_price + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TIME-BASED EXIT (10 days max hold) ===
        if in_position and (i - entry_bar) > 10:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if weekly trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend:
                new_signal = 0.0
            if position_side < 0 and bull_trend:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_bar = 0
        
        signals[i] = new_signal
    
    return signals