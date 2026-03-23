#!/usr/bin/env python3
"""
Experiment #808: 30m Primary + 4h/1d HTF — Regime-Adaptive KAMA + Fisher Entry

Hypothesis: After 550+ failed experiments, the key insight is:
1. 30m needs VERY strict filters to avoid fee drag (target 40-80 trades/year)
2. 1d Choppiness cleanly separates bull/bear/range regimes
3. 4h HMA(21) provides stable trend bias without lag
4. KAMA(14) adapts to volatility — fast in trends, slow in ranges
5. Ehlers Fisher Transform(9) catches reversals better than RSI in bear markets
6. Session filter (8-20 UTC) avoids low-liquidity Asian hours whipsaws
7. Volume confirmation (>0.8x 20-bar avg) ensures real moves

Strategy design:
1. 1d Choppiness(14) for regime: >55=range, <45=trend
2. 4h HMA(21) for trend bias (aligned via mtf_data)
3. 30m KAMA(14) for adaptive trend following
4. 30m Fisher(9) for entry timing (crosses -1.5/+1.5)
5. 30m ATR(14) for trailing stop (2.5x)
6. Session filter: only trade 8-20 UTC (high liquidity)
7. Volume filter: volume > 0.8 * SMA20(volume)
8. Discrete signals: 0.0, ±0.20, ±0.25

Key differences from failed 30m strategies:
- Fisher Transform instead of RSI (better for reversals)
- KAMA instead of EMA (adaptive to volatility)
- 1d regime filter (not 4h) — more stable regime detection
- Session + volume filters (reduces false signals)
- Relaxed Fisher thresholds (-1.5/+1.5 not -2/+2) for more trades

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_fisher_chop_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

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

def calculate_kama(close, period=14, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts to market noise — fast in trends, slow in ranges.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow:
        return kama
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform.
    Normalizes price to -1 to +1 range, highlights reversals.
    Entry: Fisher crosses above -1.5 (long), below +1.5 (short)
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0
            fisher_signal[i] = fisher[i-1] if i > 0 else 0
            continue
        
        # Normalized price
        x = (2 * (high[i] + low[i]) / 2 - (highest + lowest)) / (highest - lowest)
        x = np.clip(x, -0.999, 0.999)  # Prevent log domain error
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Signal line (1-bar lag)
        fisher_signal[i] = fisher[i-1] if i > 0 else 0
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 55/45 for more regime switches.
    """
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

def get_hour_from_timestamp(open_time):
    """Extract UTC hour from Binance timestamp (milliseconds)."""
    return (open_time // 1000 // 3600) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    kama_30m = calculate_kama(close, period=14, fast=2, slow=30)
    fisher_30m, fisher_signal_30m = calculate_fisher_transform(high, low, period=9)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma20_30m = calculate_sma(volume, 20)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_30m[i]) or np.isnan(fisher_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if np.isnan(chop_30m[i]) or np.isnan(vol_sma20_30m[i]):
            continue
        if atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_timestamp(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma20_30m[i]
        
        # === REGIME DETECTION (1d Choppiness) ===
        ranging_regime = chop_1d_aligned[i] > 55
        trending_regime = chop_1d_aligned[i] < 45
        
        # === TREND BIAS (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === KAMA TREND (30m) ===
        kama_bullish = close[i] > kama_30m[i]
        kama_bearish = close[i] < kama_30m[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_30m[i] < -1.5
        fisher_overbought = fisher_30m[i] > 1.5
        fisher_cross_up = fisher_30m[i] > fisher_signal_30m[i] and fisher_signal_30m[i] < -1.0
        fisher_cross_down = fisher_30m[i] < fisher_signal_30m[i] and fisher_signal_30m[i] > 1.0
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (1d CHOP < 45) ===
        if trending_regime and in_session and volume_ok:
            # Long: 4h bullish + KAMA bullish + Fisher cross up from oversold
            if trend_4h_bullish and kama_bullish and fisher_cross_up:
                desired_signal = BASE_SIZE
            
            # Short: 4h bearish + KAMA bearish + Fisher cross down from overbought
            if trend_4h_bearish and kama_bearish and fisher_cross_down:
                desired_signal = -BASE_SIZE
            
            # Pullback entry in trend
            if trend_4h_bullish and kama_bullish and fisher_oversold:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if trend_4h_bearish and kama_bearish and fisher_overbought:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === RANGING REGIME LOGIC (1d CHOP > 55) ===
        elif ranging_regime and in_session and volume_ok:
            # Mean reversion: Fisher extremes only
            if fisher_oversold and trend_4h_bullish:
                desired_signal = REDUCED_SIZE
            
            if fisher_overbought and trend_4h_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: require all 3 confluence
            if trend_4h_bullish and kama_bullish and fisher_cross_up and in_session:
                desired_signal = REDUCED_SIZE
            
            if trend_4h_bearish and kama_bearish and fisher_cross_down and in_session:
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
                # Hold long if trend intact and Fisher not overbought
                if trend_4h_bullish and fisher_30m[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and Fisher not oversold
                if trend_4h_bearish and fisher_30m[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses
            if trend_4h_bearish and fisher_30m[i] > 1.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses
            if trend_4h_bullish and fisher_30m[i] < -1.0:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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