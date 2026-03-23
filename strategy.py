#!/usr/bin/env python3
"""
Experiment #674: 4h Primary + 12h/1d HTF — HMA Trend + Donchian Breakout + RSI Pullback

Hypothesis: 4h timeframe with 12h trend filter and 1d regime provides optimal balance
between signal quality and trade frequency. Key innovations vs failed strategies:
1. HMA(21/48) crossover for trend — proven in best 4h strategy (Sharpe 0.612)
2. Donchian(20) breakout for entry timing — worked on SOL (Sharpe 0.782-0.879)
3. RSI(14) pullback filter — ensures entries on retracements, not chasing
4. 12h HMA for macro bias — prevents counter-trend trades
5. 1d ADX for regime — ADX>25=trend, ADX<20=range (simpler than Choppiness)
6. Time-based exit — exit after 15 bars if no profit (prevents stale positions)
7. LOOSE thresholds (RSI 30/70, not 25/75) to ensure ≥10 trades/symbol

Why this should work where others failed:
- 4h TF = ~25-45 trades/year (optimal fee/signal balance per Rule 10)
- HMA + Donchian combo worked in best 4h strategy (Sharpe 0.612)
- Simpler regime (ADX vs Choppiness) = fewer conflicting conditions
- Time exit prevents drawdown from stale positions
- Position size 0.25-0.30 with 2.5x ATR trailing stop

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_rsi_12h1d_v1"
timeframe = "4h"
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
    """Relative Strength Index with proper min_periods."""
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
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_raw = 100 - (100 / (1 + rs))
        rsi[period:] = rsi_raw[period - 1:]
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout detection."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, lower
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period * 2 - 1:] = adx_raw[period * 2 - 1:]
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    hma_21_4h = calculate_hma(close, period=21)
    hma_48_4h = calculate_hma(close, period=48)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    donch_upper_4h, donch_lower_4h = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF (12h) indicators
    hma_21_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h_raw)
    
    # Calculate and align HTF (1d) indicators
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    hma_21_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    MAX_BARS_IN_TRADE = 15  # Time-based exit
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(hma_21_4h[i]) or np.isnan(hma_48_4h[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(donch_upper_4h[i]) or np.isnan(donch_lower_4h[i]):
            continue
        if np.isnan(hma_21_12h_aligned[i]):
            continue
        if np.isnan(adx_1d_aligned[i]) or np.isnan(hma_21_1d_aligned[i]):
            continue
        
        # === 12H MACRO BIAS ===
        macro_bullish = close[i] > hma_21_12h_aligned[i]
        macro_bearish = close[i] < hma_21_12h_aligned[i]
        
        # === 1D REGIME (ADX) ===
        adx_value = adx_1d_aligned[i]
        is_trend_regime = adx_value > 25
        is_range_regime = adx_value < 20
        
        # === 1D TREND FILTER ===
        daily_bullish = close[i] > hma_21_1d_aligned[i]
        daily_bearish = close[i] < hma_21_1d_aligned[i]
        
        # === 4H TREND (HMA Crossover) ===
        hma_bullish = hma_21_4h[i] > hma_48_4h[i]
        hma_bearish = hma_21_4h[i] < hma_48_4h[i]
        hma_slope_up = hma_21_4h[i] > hma_21_4h[i - 5] if i >= 5 else False
        hma_slope_down = hma_21_4h[i] < hma_21_4h[i - 5] if i >= 5 else False
        
        # === RSI PULLBACK FILTER (LOOSE thresholds) ===
        rsi_oversold = rsi_4h[i] < 40
        rsi_overbought = rsi_4h[i] > 60
        rsi_neutral = 35 <= rsi_4h[i] <= 65
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donch_upper_4h[i - 1] if i > 0 else False
        donchian_breakout_short = close[i] < donch_lower_4h[i - 1] if i > 0 else False
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        if macro_bullish and daily_bullish:
            # Trend regime: HMA bullish + Donchian breakout + RSI not overbought
            if is_trend_regime and hma_bullish:
                if donchian_breakout_long and rsi_4h[i] < 65:
                    desired_signal = SIZE_LONG
                # Pullback entry: HMA bullish + RSI oversold
                elif hma_slope_up and rsi_oversold and close[i] > hma_21_4h[i]:
                    desired_signal = SIZE_LONG
            
            # Range regime: Mean reversion at Donchian lower
            elif is_range_regime:
                if rsi_oversold and close[i] < donch_lower_4h[i] * 1.005:
                    desired_signal = SIZE_LONG * 0.7  # Smaller size in range
        
        # === SHORT ENTRY CONDITIONS ===
        elif macro_bearish and daily_bearish:
            # Trend regime: HMA bearish + Donchian breakout + RSI not oversold
            if is_trend_regime and hma_bearish:
                if donchian_breakout_short and rsi_4h[i] > 35:
                    desired_signal = -SIZE_SHORT
                # Pullback entry: HMA bearish + RSI overbought
                elif hma_slope_down and rsi_overbought and close[i] < hma_21_4h[i]:
                    desired_signal = -SIZE_SHORT
            
            # Range regime: Mean reversion at Donchian upper
            elif is_range_regime:
                if rsi_overbought and close[i] > donch_upper_4h[i] * 0.995:
                    desired_signal = -SIZE_SHORT * 0.7  # Smaller size in range
        
        # === TRANSITION REGIME (20 <= ADX <= 25) ===
        else:
            # Use HMA direction with RSI filter
            if hma_bullish and rsi_4h[i] < 55 and macro_bullish:
                desired_signal = SIZE_LONG * 0.5
            elif hma_bearish and rsi_4h[i] > 45 and macro_bearish:
                desired_signal = -SIZE_SHORT * 0.5
        
        # === TIME-BASED EXIT ===
        if in_position and i - entry_bar > MAX_BARS_IN_TRADE:
            # Check if profitable
            if position_side > 0 and close[i] > entry_price * 1.01:
                desired_signal = 0.0  # Take profit on time exit
            elif position_side < 0 and close[i] < entry_price * 0.99:
                desired_signal = 0.0  # Take profit on time exit
            else:
                desired_signal = 0.0  # Exit at breakeven or small loss
        
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
                if hma_bullish and rsi_4h[i] < 70:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HMA still bearish AND RSI not extremely oversold
                if hma_bearish and rsi_4h[i] > 30:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.15:
            desired_signal = SIZE_LONG
        elif desired_signal < -0.15:
            desired_signal = -SIZE_SHORT
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                entry_bar = i
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
                entry_bar = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals