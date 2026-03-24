#!/usr/bin/env python3
"""
Experiment #666: 12h Primary + 1d HTF — Dual Regime with HMA + RSI + Donchian

Hypothesis: 12h timeframe with daily HTF filter provides optimal balance between
signal quality and trade frequency. Dual regime (trend vs mean-revert) based on
ADX + Bollinger Width adapts to market conditions. Key innovation: LOOSER entry
thresholds to ensure adequate trade generation (learned from 441 failures).

Why this should work:
1. 12h TF = ~30-50 trades/year (sweet spot for fee drag vs signal quality)
2. 1d HMA for macro bias — prevents counter-trend trades
3. ADX(14) > 25 = trend regime, ADX < 20 = mean-revert regime
4. BB Width percentile for regime confirmation (narrow = squeeze breakout coming)
5. RSI(14) with LOOSE thresholds (30/70 not 20/80) to ensure trades trigger
6. Donchian(20) breakout for trend entries
7. Trailing ATR stoploss (2.5x) to protect capital

Key lessons from failures:
- CRSI strategies generated 0 trades (too strict)
- Choppiness Index alone doesn't work
- Need LOOSER thresholds to ensure trade generation
- ALL symbols must have positive Sharpe (no SOL-only strategies)

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_hma_rsi_donchian_1d_v1"
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
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        adx[period*2-1:] = adx_raw[period*2-1:]
    
    return adx

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with width for regime detection."""
    n = len(close)
    bb_mid = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    bb_pct = np.full(n, np.nan)
    
    if n < period:
        return bb_mid, bb_upper, bb_lower, bb_width, bb_pct
    
    # Rolling mean and std
    bb_mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    bb_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    bb_upper = bb_mid + std_mult * bb_std
    bb_lower = bb_mid - std_mult * bb_std
    
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_width = (bb_upper - bb_lower) / bb_mid
        bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    return bb_mid, bb_upper, bb_lower, bb_width, bb_pct

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(close) if 'close' in dir() else len(high)
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    return donchian_upper, donchian_lower, donchian_mid

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
    bb_mid, bb_upper, bb_lower, bb_width, bb_pct = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # BB Width percentile for regime (narrow = squeeze)
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x[-1]) / len(x), raw=True
    ).values
    
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
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(hma_12h[i]) or np.isnan(rsi_12h[i]):
            continue
        if np.isnan(adx_12h[i]) or np.isnan(bb_width[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION ===
        adx_value = adx_12h[i]
        is_trend_regime = adx_value > 22  # Trending
        is_chop_regime = adx_value < 18   # Choppy/mean-revert
        
        # BB Width squeeze detection
        is_squeeze = bb_width_pct[i] < 0.2 if not np.isnan(bb_width_pct[i]) else False
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === 12h TREND (HMA) ===
        hma_bullish = close[i] > hma_12h[i]
        hma_bearish = close[i] < hma_12h[i]
        
        # === RSI SIGNALS (LOOSE thresholds for trade generation) ===
        rsi_oversold = rsi_12h[i] < 35
        rsi_overbought = rsi_12h[i] > 65
        rsi_neutral = 35 <= rsi_12h[i] <= 65
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === BB POSITION ===
        bb_near_lower = bb_pct[i] < 0.15 if not np.isnan(bb_pct[i]) else False
        bb_near_upper = bb_pct[i] > 0.85 if not np.isnan(bb_pct[i]) else False
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (ADX > 22) — Trend Follow ===
        if is_trend_regime:
            # Long: HTF bullish + 12h HMA bullish + RSI not overbought + Donchian breakout OR pullback
            if htf_1d_bullish and hma_bullish:
                if donchian_breakout_long or (rsi_oversold and rsi_12h[i] > 25):
                    desired_signal = SIZE_LONG
                elif rsi_neutral and close[i] > hma_12h[i] * 0.98:
                    desired_signal = SIZE_LONG
            
            # Short: HTF bearish + 12h HMA bearish + RSI not oversold
            elif htf_1d_bearish and hma_bearish:
                if donchian_breakout_short or (rsi_overbought and rsi_12h[i] < 75):
                    desired_signal = -SIZE_SHORT
                elif rsi_neutral and close[i] < hma_12h[i] * 1.02:
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 2: CHOPPY (ADX < 18) — Mean Reversion ===
        elif is_chop_regime:
            # Long: RSI oversold + BB near lower + HTF not strongly bearish
            if rsi_oversold and bb_near_lower and not htf_1d_bearish:
                desired_signal = SIZE_LONG
            # Short: RSI overbought + BB near upper + HTF not strongly bullish
            elif rsi_overbought and bb_near_upper and not htf_1d_bullish:
                desired_signal = -SIZE_SHORT
            # Squeeze breakout
            elif is_squeeze:
                if donchian_breakout_long and htf_1d_bullish:
                    desired_signal = SIZE_LONG
                elif donchian_breakout_short and htf_1d_bearish:
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 3: TRANSITION (18 <= ADX <= 22) — Mixed ===
        else:
            # Use HMA direction with RSI filter
            if hma_bullish and rsi_12h[i] < 60:
                desired_signal = SIZE_LONG
            elif hma_bearish and rsi_12h[i] > 40:
                desired_signal = -SIZE_SHORT
        
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish AND RSI not extremely overbought
                if hma_bullish and rsi_12h[i] < 75:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HMA still bearish AND RSI not extremely oversold
                if hma_bearish and rsi_12h[i] > 25:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
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