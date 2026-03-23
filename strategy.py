#!/usr/bin/env python3
"""
Experiment #672: 12h Primary + 1d/1w HTF — Simplified Dual Regime with HMA + RSI

Hypothesis: Previous dual-regime strategies failed due to overly complex logic and
too-strict regime thresholds. This version simplifies to: HTF trend bias (1d HMA)
determines direction, 12h RSI pullbacks determine entry timing, ADX filters extreme
chop. Key changes from #666:
1. LOOSER ADX thresholds (15/25 not 18/22) to ensure trades trigger
2. Simpler entry logic — HTF bias + RSI extreme = entry (no Donchian requirement)
3. Signal persistence — hold positions through minor fluctuations
4. Better stoploss — 2.5x ATR trailing stop with signal→0 on breach
5. Ensure signals array is fully populated (no gaps from continue statements)

Why 12h works: ~30-50 trades/year sweet spot, captures multi-day swings without
noise of lower TFs. 1d HTF prevents counter-trend trades in strong moves.

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_adx_regime_1d_v2"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi_raw = 100 - (100 / (1 + rs))
        rsi[period:] = rsi_raw[period-1:]
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        adx[period*2-1:] = adx_raw[period*2-1:]
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average for major trend filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    hma_12h = calculate_hma(close, period=21)
    rsi_12h = calculate_rsi(close, period=14)
    adx_12h = calculate_adx(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF indicators (1d HMA for bias)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start later to ensure all indicators ready
        # Check indicator availability
        hma_12h_valid = not np.isnan(hma_12h[i])
        rsi_12h_valid = not np.isnan(rsi_12h[i])
        adx_12h_valid = not np.isnan(adx_12h[i])
        atr_12h_valid = not np.isnan(atr_12h[i]) and atr_12h[i] > 1e-10
        hma_1d_valid = not np.isnan(hma_1d_aligned[i])
        sma_200_valid = not np.isnan(sma_200[i])
        
        if not all([hma_12h_valid, rsi_12h_valid, adx_12h_valid, atr_12h_valid, hma_1d_valid]):
            # Keep previous signal if in position, otherwise 0
            if in_position:
                signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === 12h TREND (HMA) ===
        hma_bullish = close[i] > hma_12h[i]
        hma_bearish = close[i] < hma_12h[i]
        
        # === SMA 200 MAJOR TREND ===
        above_sma200 = close[i] > sma_200[i] if sma_200_valid else True
        below_sma200 = close[i] < sma_200[i] if sma_200_valid else False
        
        # === RSI SIGNALS (LOOSE thresholds for trade generation) ===
        rsi_oversold = rsi_12h[i] < 40
        rsi_overbought = rsi_12h[i] > 60
        rsi_neutral = 40 <= rsi_12h[i] <= 60
        
        # === ADX REGIME ===
        adx_value = adx_12h[i]
        is_trending = adx_value > 20
        is_choppy = adx_value < 15
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: HTF bullish + 12h HMA bullish + RSI pullback (not overbought)
        if htf_bullish and hma_bullish:
            if rsi_12h[i] < 55 and above_sma200:
                desired_signal = SIZE_LONG
            elif rsi_oversold and htf_bullish:
                # Strong oversold in HTF bullish = good long
                desired_signal = SIZE_LONG
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: HTF bearish + 12h HMA bearish + RSI pullback (not oversold)
        elif htf_bearish and hma_bearish:
            if rsi_12h[i] > 45 and below_sma200:
                desired_signal = -SIZE_SHORT
            elif rsi_overbought and htf_bearish:
                # Strong overbought in HTF bearish = good short
                desired_signal = -SIZE_SHORT
        
        # === CHOPPY REGIME (ADX < 15) — Mean Reversion ===
        if is_choppy and desired_signal == 0.0:
            # In chop, trade RSI extremes against recent moves
            if rsi_oversold and hma_bullish:
                desired_signal = SIZE_LONG
            elif rsi_overbought and hma_bearish:
                desired_signal = -SIZE_SHORT
        
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish OR 12h HMA still bullish
                if htf_bullish or (hma_bullish and rsi_12h[i] < 70):
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish OR 12h HMA still bearish
                if htf_bearish or (hma_bearish and rsi_12h[i] > 30):
                    desired_signal = -SIZE_SHORT
        
        # === EXIT CONDITIONS ===
        # Long exit: HTF turns bearish + RSI overbought
        if in_position and position_side > 0:
            if htf_bearish and rsi_12h[i] > 70:
                desired_signal = 0.0
        
        # Short exit: HTF turns bullish + RSI oversold
        if in_position and position_side < 0:
            if htf_bullish and rsi_12h[i] < 30:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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
        
        signals[i] = desired_signal
    
    return signals