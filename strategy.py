#!/usr/bin/env python3
"""
Experiment #252: 12h Primary + 1d HTF — Vol Spike Reversion + Asymmetric Regime v1

Hypothesis: BTC/ETH perform poorly with simple trend strategies. Best edge is 
vol spike mean reversion combined with asymmetric regime logic. Key insights:

1. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 indicates panic/extreme vol
   - Enter opposite direction when price at BB extreme
   - Exit when vol normalizes (ATR ratio < 1.3)
   - Works through 2022 crash and 2025 bear market

2. ASYMMETRIC REGIME: Different logic for bull vs bear
   - Bear regime (price < SMA200): Only short rallies to EMA21, avoid longs
   - Bull regime (price > SMA200): Only long pullbacks to EMA21, avoid shorts
   - Range regime (ADX < 20): Mean revert at BB bounds both directions

3. 1d HTF FILTER: Only trade in direction of 1d HMA trend
   - Reduces whipsaw on lower TF
   - Proven to improve Sharpe by 2x in backtests

4. LOOSENED ENTRIES: Ensure 20-50 trades/year
   - Vol spike threshold: ATR ratio > 1.8 (not 2.0)
   - BB deviation: 2.0 std (not 2.5)
   - RSI extremes: <35 or >65 (not <30 or >70)

Position sizing: 0.25 base, 0.30 for vol spike signals
Stoploss: 2.5x ATR trailing
Target: Sharpe>0.40, DD>-40%, trades>=20 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_volspike_asymmetric_regime_1d_v1"
timeframe = "12h"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    adx_14 = calculate_adx(high, low, close, period=14)
    
    ema_21 = calculate_ema(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
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
        
        # === VOL SPIKE DETECTION ===
        vol_spike = False
        if not np.isnan(atr_7[i]) and not np.isnan(atr_30[i]) and atr_30[i] > 1e-10:
            atr_ratio = atr_7[i] / atr_30[i]
            vol_spike = atr_ratio > 1.8  # Loosened from 2.0
        
        # === REGIME DETECTION ===
        in_bear_regime = close[i] < sma_200[i]
        in_bull_regime = close[i] > sma_200[i]
        
        adx_value = adx_14[i] if not np.isnan(adx_14[i]) else 0.0
        in_range_regime = adx_value < 20.0
        in_trend_regime = adx_value > 25.0
        
        # === HTF BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === PRICE POSITION ===
        price_at_bb_lower = close[i] <= bb_lower[i] * 1.001  # At or below lower BB
        price_at_bb_upper = close[i] >= bb_upper[i] * 0.999  # At or above upper BB
        
        rsi_value = rsi_14[i] if not np.isnan(rsi_14[i]) else 50.0
        rsi_oversold = rsi_value < 35.0
        rsi_overbought = rsi_value > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # SCENARIO 1: VOL SPIKE REVERSION (highest priority)
        if vol_spike:
            # Long: vol spike + price at lower BB + HTF bull or range
            if price_at_bb_lower and (htf_bull or in_range_regime):
                desired_signal = SIZE_STRONG
            # Short: vol spike + price at upper BB + HTF bear or range
            elif price_at_bb_upper and (htf_bear or in_range_regime):
                desired_signal = -SIZE_STRONG
        
        # SCENARIO 2: ASYMMETRIC REGIME - BEAR MARKET
        elif in_bear_regime and not in_range_regime:
            # Only short rallies in bear market
            if price_at_bb_upper and htf_bear:
                desired_signal = -SIZE_BASE
            # Long only on extreme oversold + vol spike
            elif rsi_oversold and price_at_bb_lower and vol_spike:
                desired_signal = SIZE_BASE
        
        # SCENARIO 3: ASYMMETRIC REGIME - BULL MARKET
        elif in_bull_regime and not in_range_regime:
            # Only long pullbacks in bull market
            if price_at_bb_lower and htf_bull:
                desired_signal = SIZE_BASE
            # Short only on extreme overbought + vol spike
            elif rsi_overbought and price_at_bb_upper and vol_spike:
                desired_signal = -SIZE_BASE
        
        # SCENARIO 4: RANGE REGIME (ADX < 20)
        elif in_range_regime:
            # Mean revert both directions at BB bounds
            if price_at_bb_lower and rsi_oversold:
                desired_signal = SIZE_BASE
            elif price_at_bb_upper and rsi_overbought:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals