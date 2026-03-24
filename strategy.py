#!/usr/bin/env python3
"""
Experiment #163: 6h Primary + 1d/1w HTF — Fisher Transform + Vol Spike Reversion + Donchian Bias

Hypothesis: 6h timeframe offers optimal trade frequency (30-60/year) between 4h fee drag
and 12h too-few-trades. Previous 6h strategies failed due to overly complex regime logic
or zero trades from strict conditions.

Key insight from market analysis:
- BTC 2025+ is bear/range market (-25%), not bull trend
- Simple trend following fails on BTC/ETH (200+ failed experiments prove this)
- Mean reversion + vol spike reversion works better in bear/range regimes
- Fisher Transform excels at catching reversals in bear market rallies
- ATR ratio extremes (>2.0) signal panic capitulation → reversal opportunity

Strategy design:
- 6h Fisher Transform (period=9) for reversal timing
- 6h ATR ratio (ATR7/ATR30) for vol spike detection (>2.0 = extreme)
- 1d Donchian(20) for major trend bias (price vs 20-day range)
- 1w HMA(50) for weekly regime filter (only trade with weekly trend)
- RSI(14) loose confirmation (>40 long, <60 short) to ensure trades
- 2.5x ATR trailing stop for risk management
- Position size: 0.30 (30% of capital) with discrete levels

Trade generation safeguards (CRITICAL - avoid 0 trades):
- Multiple entry paths with decreasing size requirements
- Fallback entries when HTF strongly aligned
- Loose RSI thresholds (40/60 not 30/70)
- Vol spike entry (panic reversal) as independent signal path

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_volspike_donchian_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    Reference: Ehlers, J.F. "Cycle Analytics for Traders"
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate price position within recent range
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > period else 0.0
            fisher_prev[i] = fisher[i-1] if i > period else 0.0
            continue
        
        # Normalize price to 0-1 range
        price_norm = (close[i] - lowest) / range_val
        
        # Clamp to avoid extreme values
        price_norm = max(0.001, min(0.999, price_norm))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + price_norm) / (1.0 - price_norm))
        
        # Smooth with previous value (alpha=0.67)
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for volatility spike detection"""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    n = len(close)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

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

def calculate_donchian_channels(high, low, period=20):
    """Donchian Channels - upper/lower bounds of N-period range"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d Donchian for trend bias
    donchian_1d_upper, donchian_1d_lower, donchian_1d_mid = calculate_donchian_channels(
        df_1d['high'].values, df_1d['low'].values, period=20
    )
    donchian_1d_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_1d_upper)
    donchian_1d_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_1d_lower)
    
    # Calculate and align 1w HMA for weekly regime filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_1d_upper_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY REGIME (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1D DONCHIAN TREND BIAS ===
        # Price in upper half of 20-day range = bullish bias
        donchian_range = donchian_1d_upper_aligned[i] - donchian_1d_lower_aligned[i]
        if donchian_range > 1e-10:
            donchian_position = (close[i] - donchian_1d_lower_aligned[i]) / donchian_range
        else:
            donchian_position = 0.5
        
        donchian_bull = donchian_position > 0.5
        donchian_bear = donchian_position < 0.5
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR ratio > 2.0 = extreme volatility (panic capitulation)
        vol_spike = atr_ratio[i] > 2.0
        vol_normal = atr_ratio[i] < 1.5
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Fisher crosses above -1.5 from below = bullish reversal
        # Fisher crosses below +1.5 from above = bearish reversal
        fisher_bull_cross = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_bear_cross = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extreme oversold/overbought
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === RSI CONFIRMATION (LOOSE to ensure trades) ===
        rsi_ok_long = rsi[i] > 40.0
        rsi_ok_short = rsi[i] < 60.0
        rsi_oversold = rsi[i] < 45.0
        rsi_overbought = rsi[i] > 55.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PATH 1: Vol Spike Reversal (highest priority - panic capitulation)
        # Long: vol spike + fisher oversold + rsi oversold
        if vol_spike and fisher_oversold and rsi_oversold:
            desired_signal = SIZE
        
        # Short: vol spike + fisher overbought + rsi overbought
        elif vol_spike and fisher_overbought and rsi_overbought:
            desired_signal = -SIZE
        
        # PATH 2: Fisher Cross + HTF Alignment
        # Long: fisher bull cross + weekly bull + donchian bull + rsi ok
        elif fisher_bull_cross and htf_1w_bull and donchian_bull and rsi_ok_long:
            desired_signal = SIZE * 0.9
        
        # Short: fisher bear cross + weekly bear + donchian bear + rsi ok
        elif fisher_bear_cross and htf_1w_bear and donchian_bear and rsi_ok_short:
            desired_signal = -SIZE * 0.9
        
        # PATH 3: Fisher Cross + 1d Donchian Only (ignore weekly)
        # Long: fisher bull cross + donchian bull + rsi ok + vol normal
        elif fisher_bull_cross and donchian_bull and rsi_ok_long and vol_normal:
            desired_signal = SIZE * 0.7
        
        # Short: fisher bear cross + donchian bear + rsi ok + vol normal
        elif fisher_bear_cross and donchian_bear and rsi_ok_short and vol_normal:
            desired_signal = -SIZE * 0.7
        
        # PATH 4: Fisher Extreme Mean Reversion (no HTF filter)
        # Ensures trades even in choppy markets
        # Long: fisher < -2.0 (extreme oversold)
        elif fisher[i] < -2.0 and rsi[i] < 50.0:
            desired_signal = SIZE * 0.5
        
        # Short: fisher > 2.0 (extreme overbought)
        elif fisher[i] > 2.0 and rsi[i] > 50.0:
            desired_signal = -SIZE * 0.5
        
        # PATH 5: Weekly Trend Fallback (strong HTF alignment)
        # Long: weekly bull + donchian bull + fisher > -1.0 (not extreme bear)
        elif htf_1w_bull and donchian_bull and fisher[i] > -1.0 and rsi[i] > 45.0:
            desired_signal = SIZE * 0.6
        
        # Short: weekly bear + donchian bear + fisher < 1.0 (not extreme bull)
        elif htf_1w_bear and donchian_bear and fisher[i] < 1.0 and rsi[i] < 55.0:
            desired_signal = -SIZE * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.95:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.95:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.8:
            final_signal = SIZE * 0.9
        elif desired_signal <= -SIZE * 0.8:
            final_signal = -SIZE * 0.9
        elif desired_signal >= SIZE * 0.6:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.6:
            final_signal = -SIZE * 0.7
        elif desired_signal >= SIZE * 0.4:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.4:
            final_signal = -SIZE * 0.5
        elif desired_signal >= SIZE * 0.2:
            final_signal = SIZE * 0.3
        elif desired_signal <= -SIZE * 0.2:
            final_signal = -SIZE * 0.3
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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
        
        signals[i] = final_signal
    
    return signals