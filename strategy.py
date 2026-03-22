#!/usr/bin/env python3
"""
Experiment #459: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After analyzing 445+ failed experiments, clear pattern emerges:
1. Complex regime switching (CRSI+CHOP) leads to 0 trades or negative Sharpe
2. Simpler breakout strategies worked best historically (SOL Sharpe +0.879)
3. 4h timeframe is proven to work (20-50 trades/year target)
4. 1d HMA provides clean trend bias without over-filtering
5. Donchian(20) breakout catches major moves without whipsaw
6. RSI(14) > 50/< 50 is simpler and more reliable than extreme CRSI

Why this might beat current best (Sharpe=0.435):
- Fewer conflicting filters = more trades = better statistical significance
- Donchian breakout proven in research notes for SOL/ETH
- 1d HMA trend filter prevents counter-trend trades
- ATR 2.5x trailing stop protects in crashes
- Asymmetric sizing (0.30 long, 0.25 short) for bear market protection

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_1d_trend_v1"
timeframe = "4h"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    
    return upper, lower, mid

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # 4h HMA for local trend
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d HMA = bull bias (favor longs)
        # Price below 1d HMA = bear bias (favor shorts)
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL TREND ===
        hma_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === RSI MOMENTUM ===
        rsi_bullish = rsi_14[i] > 50.0
        rsi_bearish = rsi_14[i] < 50.0
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout above upper channel
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        # Breakout below lower channel
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_14[i] > 25.0
        weak_trend = adx_14[i] < 20.0
        
        # === ENTRY LOGIC — SIMPLIFIED FOR MORE TRADES ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple confluence, but not too strict)
        if bull_regime:
            # Donchian breakout + RSI confirmation + trend alignment
            if breakout_long and rsi_bullish and hma_bullish:
                new_signal = LONG_SIZE
            # HMA crossover + RSI confirmation (pullback entry)
            elif hma_bullish and rsi_14[i] > 55.0 and close[i] > hma_4h_21[i]:
                new_signal = LONG_SIZE * 0.8
            # Simple momentum entry in bull regime
            elif rsi_14[i] > 60.0 and hma_bullish and strong_trend:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (multiple confluence, but not too strict)
        if bear_regime:
            # Donchian breakdown + RSI confirmation + trend alignment
            if breakout_short and rsi_bearish and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # HMA crossover + RSI confirmation (bounce entry)
            elif hma_bearish and rsi_14[i] < 45.0 and close[i] < hma_4h_21[i]:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Simple momentum entry in bear regime
            elif rsi_14[i] < 40.0 and hma_bearish and strong_trend:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no position and no signal, enter on simpler conditions
        if not in_position and new_signal == 0.0:
            # Long: RSI > 55 + HMA bullish + bull regime (simpler entry)
            if bull_regime and hma_bullish and rsi_14[i] > 55.0:
                new_signal = LONG_SIZE * 0.5
            # Short: RSI < 45 + HMA bearish + bear regime (simpler entry)
            elif bear_regime and hma_bearish and rsi_14[i] < 45.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.5
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # HMA crossover exit
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals