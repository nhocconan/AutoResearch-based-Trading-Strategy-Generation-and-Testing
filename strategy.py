#!/usr/bin/env python3
"""
Experiment #997: 1d Primary + 1w HTF — KAMA Adaptive Trend + Donchian Breakout + ADX Filter

Hypothesis: After 722 failed strategies, daily timeframe with adaptive trend following
should work better than mean-reversion approaches. Key insights:

1. KAMA (Kaufman Adaptive Moving Average) adapts to volatility - faster in trends,
   slower in chop. Better than EMA/HMA for crypto's regime changes.
2. ADX(14) > 20 filters out weak trends (achievable threshold, not 40+)
3. Donchian(20) breakout - classic turtle trading signal, works on daily
4. 1w HMA(21) for macro bias - only trade in direction of weekly trend
5. ATR(14) trailing stop at 2.5x for risk management

Why 1d timeframe:
- Target 20-50 trades/year (minimal fee drag)
- Cleaner signals, less noise than lower TF
- Proven in experiment notes: Donchian+HMA+RSI+ATR worked on SOL (Sharpe +0.782)
- KAMA+ADX+Chop worked on ETH (Sharpe +0.755)

Critical improvements over failures:
- ADX threshold = 20 (not 40) to ensure trades generate
- Donchian breakout = price breaks 20-day high/low (achievable on daily)
- Weekly HMA for bias only (not strict filter)
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_donchian_adx_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength.
    ADX > 20 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
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
    
    # Calculate +DI, -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period*2:] = adx_series[period*2:]
    
    return adx

def calculate_donchian(high, low, period=20):
    """
    Donchian Channels - highest high and lowest low over period.
    Breakout above upper = long signal, below lower = short signal.
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_1d = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_1d = calculate_atr(high, low, close, period=14)
    rsi_1d = calculate_rsi(close, period=14)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d[i]) or np.isnan(adx_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(rsi_1d[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        
        # === MACRO TREND BIAS (1w HTF HMA21) ===
        macro_bullish = close[i] > hma_1w_aligned[i]
        macro_bearish = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx_1d[i] > 20  # Achievable threshold
        trend_very_strong = adx_1d[i] > 30
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        
        # Check KAMA slope (simplified - compare to 3 bars ago)
        kama_slope_up = False
        kama_slope_down = False
        if i >= 3 and not np.isnan(kama_1d[i-3]):
            kama_slope_up = kama_1d[i] > kama_1d[i-3]
            kama_slope_down = kama_1d[i] < kama_1d[i-3]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI FILTER ===
        rsi_neutral = 35 < rsi_1d[i] < 65
        rsi_bullish = rsi_1d[i] > 50
        rsi_bearish = rsi_1d[i] < 50
        rsi_not_overbought = rsi_1d[i] < 70
        rsi_not_oversold = rsi_1d[i] > 30
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: Donchian breakout + trend strength + KAMA bullish + macro bias
        if donchian_breakout_long and trend_strong and kama_bullish and macro_bullish:
            desired_signal = BASE_SIZE
        # Secondary: KAMA bullish + ADX strong + RSI bullish (no breakout needed)
        elif kama_bullish and kama_slope_up and adx_1d[i] > 25 and rsi_bullish and macro_bullish:
            desired_signal = REDUCED_SIZE
        # Tertiary: Macro bullish + KAMA crossover (price crosses above KAMA)
        elif macro_bullish and kama_bullish and close[i-1] <= kama_1d[i-1] if not np.isnan(kama_1d[i-1]) else False:
            if trend_strong and rsi_not_overbought:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: Donchian breakdown + trend strength + KAMA bearish + macro bias
        if donchian_breakout_short and trend_strong and kama_bearish and macro_bearish:
            desired_signal = -BASE_SIZE
        # Secondary: KAMA bearish + ADX strong + RSI bearish (no breakdown needed)
        elif kama_bearish and kama_slope_down and adx_1d[i] > 25 and rsi_bearish and macro_bearish:
            desired_signal = -REDUCED_SIZE
        # Tertiary: Macro bearish + KAMA crossdown (price crosses below KAMA)
        elif macro_bearish and kama_bearish and close[i-1] >= kama_1d[i-1] if not np.isnan(kama_1d[i-1]) else False:
            if trend_strong and rsi_not_oversold:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish and ADX still strong
                if kama_bullish and adx_1d[i] > 18 and rsi_not_overbought:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA still bearish and ADX still strong
                if kama_bearish and adx_1d[i] > 18 and rsi_not_oversold:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses + ADX weakens
            if kama_bearish and adx_1d[i] < 15:
                desired_signal = 0.0
            # Exit if macro trend reverses strongly
            if macro_bearish and rsi_1d[i] < 40:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses + ADX weakens
            if kama_bullish and adx_1d[i] < 15:
                desired_signal = 0.0
            # Exit if macro trend reverses strongly
            if macro_bullish and rsi_1d[i] > 60:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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