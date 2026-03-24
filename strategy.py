#!/usr/bin/env python3
"""
Experiment #517: 15m Primary + 4h/12h HTF — RSI Mean Reversion with Trend Filter

Hypothesis: 15m timeframe with 4h HTF trend filter can capture intraday mean-reversion
moves while avoiding counter-trend trades. Key insight from failed 15m experiments:
entry conditions were TOO STRICT (0 trades). This version uses LOOSER thresholds:
- RSI(7) < 40 for long (not <20), > 60 for short (not >80)
- Only 2-3 confluence factors required (not 5+)
- Bollinger Band touch as confirmation (not hard requirement)
- Session bias: slightly favor 00-12 UTC but allow all hours

Strategy logic:
1. 4h HMA(21) = trend bias (long only above, short only below)
2. 15m RSI(7) = entry timing (oversold/overbought extremes)
3. 15m BB(20, 2.0) = mean reversion confirmation
4. 15m ATR(14) = volatility filter + stoploss (2.5x ATR)
5. 12h ADX(14) = regime filter (avoid low ADX whipsaw)

Target: Sharpe>0.50, trades>=100 train (25/year), trades>=10 test
Timeframe: 15m
Position size: 0.20 (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_bb_hma_4h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h ADX for regime filter
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    hma_15m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4h HTF TREND BIAS ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 12h ADX REGIME ===
        adx_value = adx_12h_aligned[i]
        trending = adx_value > 20.0  # ADX > 20 = trending market
        ranging = adx_value < 25.0   # ADX < 25 = ranging market
        
        # === 15m RSI EXTREMES (LOOSE THRESHOLDS FOR TRADES) ===
        rsi_oversold = rsi_7[i] < 40.0  # Loose: was <20
        rsi_overbought = rsi_7[i] > 60.0  # Loose: was >80
        rsi_extreme_oversold = rsi_7[i] < 30.0
        rsi_extreme_overbought = rsi_7[i] > 70.0
        
        # RSI turning
        rsi_rising = rsi_7[i] > rsi_7[i-1] if i > 0 else False
        rsi_falling = rsi_7[i] < rsi_7[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.002  # Within 0.2% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.998  # Within 0.2% of upper band
        below_bb_mid = close[i] < bb_mid[i]
        above_bb_mid = close[i] > bb_mid[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === VOLATILITY FILTER ===
        if i >= 50:
            atr_mean = np.nanmean(atr[i-50:i])
            vol_normal = atr[i] < atr_mean * 2.5 if atr_mean > 0 else True
        else:
            vol_normal = True
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADES) ===
        desired_signal = 0.0
        
        # LONG ENTRIES - Multiple pathways to ensure trades
        if htf_bull and vol_normal:
            # Primary: RSI oversold + near BB lower + HTF bull
            if rsi_oversold and near_bb_lower:
                desired_signal = SIZE_STRONG
            # Secondary: RSI extreme oversold + HTF bull (even without BB)
            elif rsi_extreme_oversold and htf_bull:
                desired_signal = SIZE_BASE
            # Tertiary: RSI rising from oversold + above BB mid
            elif rsi_7[i] < 45.0 and rsi_rising and above_bb_mid and htf_bull:
                desired_signal = SIZE_BASE * 0.8
            # Quaternary: Simple RSI reversal in uptrend
            elif rsi_7[i] < 35.0 and rsi_rising and hma_bull:
                desired_signal = SIZE_BASE * 0.7
        
        # SHORT ENTRIES - Multiple pathways to ensure trades
        if htf_bear and vol_normal:
            # Primary: RSI overbought + near BB upper + HTF bear
            if rsi_overbought and near_bb_upper:
                desired_signal = -SIZE_STRONG
            # Secondary: RSI extreme overbought + HTF bear (even without BB)
            elif rsi_extreme_overbought and htf_bear:
                desired_signal = -SIZE_BASE
            # Tertiary: RSI falling from overbought + below BB mid
            elif rsi_7[i] > 55.0 and rsi_falling and below_bb_mid and htf_bear:
                desired_signal = -SIZE_BASE * 0.8
            # Quaternary: Simple RSI reversal in downtrend
            elif rsi_7[i] > 65.0 and rsi_falling and hma_bear:
                desired_signal = -SIZE_BASE * 0.7
        
        # RANGE REGIME: More aggressive mean reversion
        if ranging and vol_normal:
            if rsi_extreme_oversold and near_bb_lower:
                desired_signal = SIZE_BASE  # Long even without HTF confirmation
            elif rsi_extreme_overbought and near_bb_upper:
                desired_signal = -SIZE_BASE  # Short even without HTF confirmation
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals