#!/usr/bin/env python3
"""
Experiment #665: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + ADX Regime

Hypothesis: 1h timeframe with 4h trend filter and 1d regime detection provides
optimal trade frequency (30-80/year) while avoiding fee drag. Key insight from
failed experiments: session/volume filters KILL trade frequency on lower TFs.

Strategy logic:
1. 4h HMA(21) for trend direction (proven in best strategy mtf_4h_triple_regime)
2. 1h RSI(14) for entry timing — long when RSI<35 in uptrend, short when RSI>65 in downtrend
3. 1d ADX(14) for regime — ADX>25 = trend (follow), ADX<20 = range (mean revert)
4. ATR(14) trailing stoploss at 2.5x ATR
5. Hold logic: maintain position if HTF trend unchanged (reduces churn)

Why this should beat Sharpe=0.612:
- Simpler than CRSI (which failed 6+ times on 1h/30m/4h)
- NO session/volume filters (those caused 0 trades in #655, #658, #660, #664)
- Looser RSI thresholds (35/65 vs 25/75) ensures adequate trade frequency
- 4h HMA proven edge from current best strategy
- Conservative sizing (0.25) for lower TF fee management

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_adx_regime_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Initialize with SMA
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    
    rsi[period] = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 1e-10 else 100
    
    # Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        
        if avg_loss > 1e-10:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth with Wilder's method
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100 * plus_dm[i] / atr[i]
            minus_di[i] = 100 * minus_dm[i] / atr[i]
    
    # Smooth DI
    plus_di_smooth = np.zeros(n)
    minus_di_smooth = np.zeros(n)
    plus_di_smooth[period] = np.mean(plus_di[period-period+1:period+1])
    minus_di_smooth[period] = np.mean(minus_di[period-period+1:period+1])
    
    for i in range(period + 1, n):
        plus_di_smooth[i] = (plus_di_smooth[i - 1] * (period - 1) + plus_di[i]) / period
        minus_di_smooth[i] = (minus_di_smooth[i - 1] * (period - 1) + minus_di[i]) / period
    
    # DX and ADX
    dx = np.zeros(n)
    for i in range(period * 2, n):
        di_sum = plus_di_smooth[i] + minus_di_smooth[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(plus_di_smooth[i] - minus_di_smooth[i]) / di_sum
    
    # ADX = SMA of DX
    adx[period * 2] = np.mean(dx[period * 2 - period + 1:period * 2 + 1])
    for i in range(period * 2 + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother trend detection."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative for 1h timeframe
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(adx_1d_aligned[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1d ADX) ===
        adx_value = adx_1d_aligned[i]
        is_trending = adx_value > 25.0
        is_ranging = adx_value < 20.0
        
        # === RSI SIGNALS (1h) ===
        rsi_value = rsi_1h[i]
        rsi_oversold = rsi_value < 35.0
        rsi_overbought = rsi_value > 65.0
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING MARKET (Follow 4h trend on RSI pullback) ===
        if is_trending:
            # Long: 4h bullish + RSI pullback to oversold
            if htf_4h_bullish and rsi_oversold:
                desired_signal = SIZE
            # Short: 4h bearish + RSI pullback to overbought
            elif htf_4h_bearish and rsi_overbought:
                desired_signal = -SIZE
        
        # === REGIME 2: RANGING MARKET (Mean reversion at extremes) ===
        elif is_ranging:
            # Long: RSI deeply oversold (mean revert)
            if rsi_value < 30.0:
                desired_signal = SIZE
            # Short: RSI deeply overbought (mean revert)
            elif rsi_value > 70.0:
                desired_signal = -SIZE
        
        # === REGIME 3: NEUTRAL/TRANSITION (Use 4h trend only) ===
        else:
            # Long: 4h bullish + RSI not overbought
            if htf_4h_bullish and rsi_value < 60.0:
                desired_signal = SIZE
            # Short: 4h bearish + RSI not oversold
            elif htf_4h_bearish and rsi_value > 40.0:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if HTF trend unchanged ===
        # This reduces churn and fee drag while capturing trends
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h still bullish OR RSI not extremely overbought
                if htf_4h_bullish or rsi_value < 75.0:
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if 4h still bearish OR RSI not extremely oversold
                if htf_4h_bearish or rsi_value > 25.0:
                    desired_signal = -SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals