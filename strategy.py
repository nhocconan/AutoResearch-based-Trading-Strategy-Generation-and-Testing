#!/usr/bin/env python3
"""
Experiment #1090: 1h Primary + 4h/12h HTF — Regime-Adaptive with Relaxed Entries

Hypothesis: After 790+ failed experiments, the key insight is:
1. Lower TF (1h) MUST generate 30-60 trades/year — too strict = 0 trades (auto-reject)
2. Use 4h/12h HTF for DIRECTION, 1h only for ENTRY TIMING
3. Regime-adaptive: CHOP > 55 = mean revert, CHOP < 45 = trend follow
4. RELAXED entry thresholds vs failed #1080/#1085: RSI 35-55 (not extremes), ADX > 18 (not 25)
5. Session filter (8-20 UTC) + volume filter to reduce noise
6. Position size: 0.25 discrete levels, stoploss 2.5x ATR

Why this should beat Sharpe=0.612:
- Regime detection avoids trend strategies in chop (major 2022 failure mode)
- Relaxed entries ensure 30-60 trades/year (not 0 like #1080/#1085)
- HTF alignment prevents counter-trend trades
- Session/volume filters reduce false signals during low liquidity

Timeframe: 1h (primary)
HTF: 4h + 12h — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_rsi_chop_4h12h_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=7):
    """RSI with shorter period for faster signals on 1h."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    hl_range = hh - ll
    
    # CHOP formula
    mask = hl_range > 1e-10
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / hl_range[mask]) / np.log10(period)
    chop[~mask] = 50.0
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = close[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    di_sum = plus_di + minus_di
    dx = np.zeros(n)
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion levels."""
    n = len(close)
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return mid, upper, lower

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time for session filter."""
    # open_time is in milliseconds since epoch
    open_time_ms = prices["open_time"].values
    hours = ((open_time_ms // 1000) // 3600) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMAs for macro trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi = calculate_rsi(close, period=7)
    chop = calculate_choppiness(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Volume average for filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours
    hours = get_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(adx[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(bb_mid[i]) or np.isnan(vol_avg[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (12h HMA21) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA21) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (1h HMA crossover) ===
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # === REGIME DETECTION (Choppiness) ===
        choppy_regime = chop[i] > 55.0  # Range/mean revert
        trending_regime = chop[i] < 45.0  # Trend follow
        
        # === TREND STRENGTH (ADX) — relaxed threshold ===
        strong_trend = adx[i] > 18.0
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_avg[i]
        
        # === RSI SIGNALS — relaxed for more trades ===
        # Long: RSI 35-55 (pullback in uptrend, not extreme)
        rsi_long = 35.0 <= rsi[i] <= 55.0
        # Short: RSI 45-65 (rally in downtrend, not extreme)
        rsi_short = 45.0 <= rsi[i] <= 65.0
        
        # === BOLLINGER POSITION ===
        bb_long = close[i] <= bb_lower[i] * 1.005  # Near or below lower band
        bb_short = close[i] >= bb_upper[i] * 0.995  # Near or above upper band
        
        # === VOLATILITY CHECK ===
        vol_spike = atr[i] > 1.8 * np.nanmedian(atr[max(0, i-100):i]) if i > 100 else False
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        desired_signal = 0.0
        
        # === LONG ENTRY — Multiple confluence paths ===
        # Path 1: Trending regime + HTF bull + RSI pullback
        if trending_regime and htf_bull and rsi_long and in_session:
            desired_signal = current_size
        # Path 2: Choppy regime + BB lower + RSI long (mean revert)
        elif choppy_regime and bb_long and rsi_long and volume_ok:
            desired_signal = current_size
        # Path 3: Strong trend + HMA bull + ADX strong
        elif strong_trend and hma_bull and macro_bull and in_session and volume_ok:
            desired_signal = current_size
        
        # === SHORT ENTRY — Multiple confluence paths ===
        # Path 1: Trending regime + HTF bear + RSI rally
        if trending_regime and htf_bear and rsi_short and in_session:
            desired_signal = -current_size
        # Path 2: Choppy regime + BB upper + RSI short (mean revert)
        elif choppy_regime and bb_short and rsi_short and volume_ok:
            desired_signal = -current_size
        # Path 3: Strong trend + HMA bear + ADX strong
        elif strong_trend and hma_bear and macro_bear and in_session and volume_ok:
            desired_signal = -current_size
        
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
                # Hold long if HTF still bullish or RSI not overbought
                if htf_bull and rsi[i] < 70.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if HTF still bearish or RSI not oversold
                if htf_bear and rsi[i] > 30.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF reverses or RSI overbought
            if htf_bear and adx[i] > 20.0:
                desired_signal = 0.0
            if rsi[i] > 70.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF reverses or RSI oversold
            if htf_bull and adx[i] > 20.0:
                desired_signal = 0.0
            if rsi[i] < 30.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals