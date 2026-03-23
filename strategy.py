#!/usr/bin/env python3
"""
Experiment #737: 1d Primary + 1w HTF — KAMA Trend + Donchian Breakout + ADX Filter

Hypothesis: After 494 failed strategies, the pattern is clear — complex regime detection
(Choppiness + CRSI) consistently fails. Simple trend-following with adaptive indicators works.

This strategy uses:
1. 1w KAMA(21) for adaptive trend bias (KAMA adjusts to volatility, better than HMA in chop)
2. 1d Donchian(20) breakout for entries (simple, generates 20-40 trades/year on 1d)
3. 1d ADX(14) > 20 for trend strength confirmation (loose threshold to ensure trades)
4. 1d ATR(14) trailing stop 2.0x for risk management
5. Discrete signal sizes: 0.0, ±0.25, ±0.30

Key improvements over #736:
- KAMA instead of HMA (adapts to market volatility, reduces whipsaw)
- ADX filter for trend strength (avoids entering in dead markets)
- 1d primary + 1w HTF (stronger trend filter than 12h+1d)
- Tighter stoploss (2.0x ATR vs 2.5x) for better risk/reward
- Simpler entry logic (fewer conflicting paths)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_donchian_adx_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility - smooth in chop, responsive in trends.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    volatility = np.zeros(n)
    for i in range(period, n):
        vol_sum = 0.0
        for j in range(1, period + 1):
            vol_sum += np.abs(close[i - j + 1] - close[i - j])
        volatility[i] = vol_sum if vol_sum > 0 else 1e-10
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/range.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    adx_1d = calculate_adx(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF KAMA for trend bias
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    take_profit_hit = False
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(adx_1d[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        
        # === TREND BIAS (1w HTF KAMA) ===
        trend_1w_bullish = close[i] > kama_1w_aligned[i]
        trend_1w_bearish = close[i] < kama_1w_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx_1d[i] > 20  # Loose threshold to ensure trades
        trend_very_strong = adx_1d[i] > 30
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI FILTERS ===
        rsi_not_overbought = rsi_1d[i] < 70
        rsi_not_oversold = rsi_1d[i] > 30
        rsi_momentum_long = rsi_1d[i] > 45
        rsi_momentum_short = rsi_1d[i] < 55
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        long_signal = False
        
        # Path 1: Donchian breakout + bullish 1w trend + ADX strong
        if close[i] > donch_upper[i-1] and trend_1w_bullish and trend_strong:
            long_signal = True
        
        # Path 2: Price above SMA50/200 + bullish 1w trend + RSI momentum
        if above_sma50 and above_sma200 and trend_1w_bullish and rsi_momentum_long:
            long_signal = True
        
        # Path 3: Strong trend (ADX > 30) + bullish 1w + price > KAMA
        if trend_very_strong and trend_1w_bullish and rsi_not_overbought:
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        short_signal = False
        
        # Path 1: Donchian breakdown + bearish 1w trend + ADX strong
        if close[i] < donch_lower[i-1] and trend_1w_bearish and trend_strong:
            short_signal = True
        
        # Path 2: Price below SMA50/200 + bearish 1w trend + RSI momentum
        if below_sma50 and below_sma200 and trend_1w_bearish and rsi_momentum_short:
            short_signal = True
        
        # Path 3: Strong trend (ADX > 30) + bearish 1w + price < KAMA
        if trend_very_strong and trend_1w_bearish and rsi_not_oversold:
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with 1w KAMA trend
        if long_signal and short_signal:
            if trend_1w_bullish:
                desired_signal = BASE_SIZE
            elif trend_1w_bearish:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (2R) ===
        if in_position and not take_profit_hit:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    take_profit_hit = True
                    desired_signal = HALF_SIZE  # Reduce to half
            elif position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    take_profit_hit = True
                    desired_signal = -HALF_SIZE  # Reduce to half
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1w KAMA still bullish and RSI not extremely overbought
                if trend_1w_bullish and rsi_1d[i] < 80:
                    desired_signal = BASE_SIZE if not take_profit_hit else HALF_SIZE
            elif position_side < 0:
                # Hold short if 1w KAMA still bearish and RSI not extremely oversold
                if trend_1w_bearish and rsi_1d[i] > 20:
                    desired_signal = -BASE_SIZE if not take_profit_hit else -HALF_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1w trend reverses or RSI extremely overbought
            if trend_1w_bearish or rsi_1d[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1w trend reverses or RSI extremely oversold
            if trend_1w_bullish or rsi_1d[i] < 15:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.20:
            desired_signal = BASE_SIZE
        elif desired_signal > 0:
            desired_signal = HALF_SIZE
        elif desired_signal < -0.20:
            desired_signal = -BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -HALF_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                take_profit_hit = False
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                take_profit_hit = False
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
                take_profit_hit = False
        
        signals[i] = desired_signal
    
    return signals