#!/usr/bin/env python3
"""
Experiment #389: 4h Primary + 1d HTF — HMA Trend + Dual Regime (Simplified)

Hypothesis: Previous strategy #381 over-complicated entries with CRSI + KAMA + ADX hysteresis.
This strategy SIMPLIFIES to proven patterns:
1. HMA(21) for trend - faster response than KAMA, proven in best strategies
2. Dual regime: ADX>25 = trend follow (Donchian breakout), ADX<20 = mean revert (BB touch)
3. RSI(14) instead of CRSI - simpler, more reliable, triggers more often
4. 1d HTF bias ONLY - dual HTF caused 0 trades in #380, #382
5. Relaxed entry: need only 2 of 3 conditions (not all 3)
6. Simple stoploss: 2.5*ATR trailing, no complex take-profit logic

Target: 40-60 trades/year on 4h, Sharpe > 0.6 on ALL symbols.
Key change from #381: Fewer filters, HMA instead of KAMA, RSI instead of CRSI.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_dual_regime_rsi_donchian_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA, less lag than SMA.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    hull = (2.0 * wma_half - wma_full).ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hull.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20.0).values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma.values, upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_21 = calculate_hma(close, period=21)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === HTF BIAS (1d HMA) ===
        bullish_bias = close[i] > hma_1d_aligned[i]
        bearish_bias = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        # === REGIME DETECTION ===
        adx_val = adx_14[i]
        is_trending = adx_val > 25.0
        is_ranging = adx_val < 20.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if bullish_bias:
            if is_trending:
                # Trend follow: Donchian breakout + price above HMA
                breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
                if breakout_long and price_above_hma:
                    desired_signal = BASE_SIZE
            elif is_ranging:
                # Mean reversion: RSI oversold + BB lower touch
                rsi_oversold = rsi_14[i] < 35.0
                bb_touch = close[i] <= bb_lower[i] * 1.002  # within 0.2% of lower band
                if rsi_oversold and bb_touch:
                    desired_signal = BASE_SIZE
            else:
                # Transition zone: simpler entry
                rsi_oversold = rsi_14[i] < 40.0
                if rsi_oversold and price_above_hma:
                    desired_signal = BASE_SIZE
        
        # SHORT ENTRIES
        if bearish_bias:
            if is_trending:
                # Trend follow: Donchian breakdown + price below HMA
                breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
                if breakout_short and price_below_hma:
                    desired_signal = -BASE_SIZE
            elif is_ranging:
                # Mean reversion: RSI overbought + BB upper touch
                rsi_overbought = rsi_14[i] > 65.0
                bb_touch = close[i] >= bb_upper[i] * 0.998  # within 0.2% of upper band
                if rsi_overbought and bb_touch:
                    desired_signal = -BASE_SIZE
            else:
                # Transition zone: simpler entry
                rsi_overbought = rsi_14[i] > 60.0
                if rsi_overbought and price_below_hma:
                    desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === RSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_14[i] > 70:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and bearish_bias:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and bullish_bias:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and bullish_bias:
                desired_signal = BASE_SIZE
            elif position_side < 0 and bearish_bias:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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