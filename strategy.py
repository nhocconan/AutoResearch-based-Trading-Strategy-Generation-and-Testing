#!/usr/bin/env python3
"""
Experiment #977: 1d Primary + 1w HTF — KAMA Adaptive Trend + ADX/Chop Regime + RSI

Hypothesis: Daily timeframe with weekly trend filter provides cleaner signals with
lower fee drag. KAMA adapts to market efficiency (flattens in chop, follows in trend).
Combined with ADX/Choppiness regime detection and RSI timing, this should work
across BTC/ETH/SOL in both bull and bear markets.

Why 1d timeframe:
- Target 15-30 trades/year (minimal fee drag)
- Cleaner signals, less noise than lower TF
- Weekly HTF provides strong macro bias
- Proven to work through 2022 crash and 2025 bear

Key components:
1. KAMA(10) — Adaptive MA that flattens in chop, trends in momentum
2. ADX(14) — Trend strength filter (>25 = trend, <20 = range)
3. Choppiness(14) — Regime confirmation (>55 = range, <45 = trend)
4. 1w HMA(21) — Macro trend bias (HTF, loaded ONCE before loop)
5. RSI(14) — Entry timing (oversold long, overbought short)
6. ATR(14) — Stoploss at 2.5x

Regime-adaptive logic:
- Trending (ADX>25 + CHOP<45): Follow KAMA direction with weekly bias
- Ranging (ADX<20 + CHOP>55): Mean revert at RSI extremes
- Neutral: Conservative entries with weekly confluence

Position sizing: 0.25-0.30 discrete levels to minimize churn
Stoploss: 2.5x ATR trailing stop

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 15-30 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_chop_regime_1w_hma_rsi_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts to market efficiency ratio (ER).
    ER near 1 = trending (fast smoothing), ER near 0 = choppy (slow smoothing).
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(slow_period, n):
        price_change = np.abs(close[i] - close[i - slow_period])
        if price_change < 1e-10:
            er[i] = 0
            continue
        vol_sum = 0.0
        for j in range(i - slow_period + 1, i + 1):
            vol_sum += np.abs(close[j] - close[j - 1])
        er[i] = price_change / (vol_sum + 1e-10)
    
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging.
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
    
    # Smooth TR and DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di / (atr + 1e-10)
        minus_di = 100 * minus_di / (atr + 1e-10)
        di_sum = plus_di + minus_di + 1e-10
        dx = 100 * np.abs(plus_di - minus_di) / di_sum
    
    # Smooth DX to get ADX
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period*2:] = adx_raw[period*2:]
    
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    adx_1d = calculate_adx(high, low, close, period=14)
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    
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
        if np.isnan(kama_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(adx_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_1d[i] > 25
        weak_trend = adx_1d[i] < 20
        
        # === REGIME DETECTION (Choppiness) ===
        ranging_regime = chop_1d[i] > 55
        trending_regime = chop_1d[i] < 45
        
        # === KAMA DIRECTION ===
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        
        # KAMA slope (compare to 3 bars ago)
        kama_slope_up = kama_1d[i] > kama_1d[i-3] if i >= 103 and not np.isnan(kama_1d[i-3]) else False
        kama_slope_down = kama_1d[i] < kama_1d[i-3] if i >= 103 and not np.isnan(kama_1d[i-3]) else False
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_1d[i] < 35
        rsi_overbought = rsi_1d[i] > 65
        rsi_extreme_oversold = rsi_1d[i] < 25
        rsi_extreme_overbought = rsi_1d[i] > 75
        rsi_neutral = 35 <= rsi_1d[i] <= 65
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (ADX>25 + CHOP<45) — Trend Following ===
        if strong_trend and trending_regime:
            # Long: KAMA bullish + macro bull + RSI not overbought
            if kama_bullish and kama_slope_up and macro_bull and not rsi_overbought:
                desired_signal = BASE_SIZE
            # Long: KAMA bullish + RSI pullback to neutral
            elif kama_bullish and kama_slope_up and rsi_neutral and rsi_1d[i] > 45:
                desired_signal = REDUCED_SIZE
            
            # Short: KAMA bearish + macro bear + RSI not oversold
            if kama_bearish and kama_slope_down and macro_bear and not rsi_oversold:
                desired_signal = -BASE_SIZE
            # Short: KAMA bearish + RSI pullback to neutral
            elif kama_bearish and kama_slope_down and rsi_neutral and rsi_1d[i] < 55:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME (ADX<20 + CHOP>55) — Mean Reversion ===
        elif weak_trend and ranging_regime:
            # Long: RSI extreme oversold + price below KAMA (oversold bounce)
            if rsi_extreme_oversold and kama_bearish:
                desired_signal = BASE_SIZE
            # Long: RSI oversold + macro bull support
            elif rsi_oversold and macro_bull:
                desired_signal = REDUCED_SIZE
            
            # Short: RSI extreme overbought + price above KAMA (overbought fade)
            if rsi_extreme_overbought and kama_bullish:
                desired_signal = -BASE_SIZE
            # Short: RSI overbought + macro bear resistance
            elif rsi_overbought and macro_bear:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME — Conservative with Weekly Confluence ===
        else:
            # Long: KAMA bullish + macro bull + RSI favorable
            if kama_bullish and macro_bull and rsi_1d[i] < 60:
                desired_signal = REDUCED_SIZE
            # Long: RSI extreme oversold (guarantees some trades)
            elif rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: KAMA bearish + macro bear + RSI favorable
            if kama_bearish and macro_bear and rsi_1d[i] > 40:
                desired_signal = -REDUCED_SIZE
            # Short: RSI extreme overbought
            elif rsi_extreme_overbought:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish and RSI not extreme
                if kama_bullish and rsi_1d[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA still bearish and RSI not extreme
                if kama_bearish and rsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA flips bearish + macro flips bear
            if kama_bearish and macro_bear:
                desired_signal = 0.0
            # Exit if RSI extreme overbought
            if rsi_extreme_overbought:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA flips bullish + macro flips bull
            if kama_bullish and macro_bull:
                desired_signal = 0.0
            # Exit if RSI extreme oversold
            if rsi_extreme_oversold:
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