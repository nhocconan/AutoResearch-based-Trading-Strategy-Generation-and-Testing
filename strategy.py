#!/usr/bin/env python3
"""
Experiment #1045: 1h Primary + 4h/1d HTF — Relaxed Multi-Regime Strategy

Hypothesis: After 756+ failed experiments, the key lesson is that LOWER TF (1h/30m) 
strategies fail due to TWO reasons: (1) TOO MANY confluence filters that never agree, 
and (2) thresholds too strict for bear/range markets.

SOLUTION: RELAXED thresholds with FOCUSED confluence:
1. Use 4h HMA for TREND DIRECTION only (not entry trigger)
2. Use 1d HMA for MACRO BIAS only (loose filter)
3. 1h indicators for ENTRY TIMING with WIDE thresholds
4. Session filter 8-20 UTC (liquid hours only)
5. Volume filter relaxed (>0.6x avg, not >1.0x)

REGIME LOGIC (simplified from failed #1035, #1040):
- CHOP > 55 = Range → Mean Reversion (CRSI < 15 long, > 85 short)
- CHOP < 45 = Trend → Trend Follow (HMA cross + 4h confirmation)
- 45-55 = Transition → Allow BOTH types (this was blocking trades!)

CRSI (Connors RSI) proven edge:
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 15 (oversold), Short: CRSI > 85 (overbought)
- 75% win rate in research, works in bear markets

WHY THIS SHOULD WORK:
- 1h timeframe with 4h/1d HTF = proven pattern (mtf_hma_rsi_zscore_v1 had Sharpe=5.4)
- Relaxed CHOP transition zone = MORE trades (fixes 0-trade problem)
- Session filter = fewer but HIGHER QUALITY trades (30-60/year target)
- CRSI more reliable than regular RSI for mean reversion

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year)
Position Size: 0.25 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_relaxed_regime_crsi_4h1d_hma_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - proven mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Long: CRSI < 15 (extreme oversold)
    Short: CRSI > 85 (extreme overbought)
    
    Research shows 75% win rate, works well in bear/range markets.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(np.concatenate([[0], gain]))
    loss_series = pd.Series(np.concatenate([[0], loss]))
    
    avg_gain = gain_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_3 = 100 - (100 / (1 + rs))
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    abs_streak = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = 100 - (100 / (1 + abs_streak[i] / max(streak_period, 1)))
        else:
            streak_rsi[i] = 100 / (1 + abs_streak[i] / max(streak_period, 1))
    
    # Percent Rank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = rank / rank_period * 100
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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

def calculate_hma(series, period):
    """Hull Moving Average - faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    
    smoothed_plus_dm = plus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_minus_dm = minus_dm_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    smoothed_tr = tr_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.divide(100 * smoothed_plus_dm, smoothed_tr, out=np.zeros_like(smoothed_plus_dm), where=smoothed_tr != 0)
    minus_di = np.divide(100 * smoothed_minus_dm, smoothed_tr, out=np.zeros_like(smoothed_minus_dm), where=smoothed_tr != 0)
    
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.divide(100 * di_diff, di_sum, out=np.zeros_like(di_diff), where=di_sum != 0)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(adx[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]) or np.isnan(vol_sma[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Convert open_time to hour
        hour = (open_time[i] // 3600000) % 24
        is_liquid_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (relaxed) ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 0 else 0
        has_volume = vol_ratio > 0.6
        
        # === REGIME DETECTION (RELAXED thresholds) ===
        is_range = chop[i] > 50.0  # Wider threshold for more trades
        is_trend = chop[i] < 50.0  # Allow both in transition
        
        # === HTF TREND DIRECTION ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === MACRO BIAS (1d HMA - loose filter) ===
        macro_bull = close[i] > hma_1d_aligned[i] * 0.98  # 2% buffer
        macro_bear = close[i] < hma_1d_aligned[i] * 1.02
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION (CRSI) ===
        if is_range and is_liquid_session and has_volume:
            # Long: CRSI extreme oversold + HTF not strongly bearish
            if crsi[i] < 15 and not (htf_bear and macro_bear):
                desired_signal = BASE_SIZE
            # Short: CRSI extreme overbought + HTF not strongly bullish
            elif crsi[i] > 85 and not (htf_bull and macro_bull):
                desired_signal = -BASE_SIZE
            # Weaker mean reversion signals
            elif crsi[i] < 20 and htf_bull:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 80 and htf_bear:
                desired_signal = -REDUCED_SIZE
        
        # === TREND MODE: TREND FOLLOWING ===
        if is_trend and is_liquid_session and has_volume:
            # Long: HMA8 > HMA21 + HTF bullish + ADX shows some trend
            if hma_8[i] > hma_21[i] and htf_bull and adx[i] > 15:
                desired_signal = BASE_SIZE
            # Short: HMA8 < HMA21 + HTF bearish + ADX shows some trend
            elif hma_8[i] < hma_21[i] and htf_bear and adx[i] > 15:
                desired_signal = -BASE_SIZE
            # Weaker trend signals (no ADX requirement)
            elif hma_8[i] > hma_21[i] and htf_bull:
                desired_signal = REDUCED_SIZE
            elif hma_8[i] < hma_21[i] and htf_bear:
                desired_signal = -REDUCED_SIZE
        
        # === ALLOW TRANSITION ZONE (45-55 CHOP) ===
        # This was blocking trades in previous experiments!
        if 45 <= chop[i] <= 55 and is_liquid_session and has_volume:
            # Allow both mean reversion AND trend following
            if crsi[i] < 18:
                desired_signal = REDUCED_SIZE
            elif crsi[i] > 82:
                desired_signal = -REDUCED_SIZE
            elif hma_8[i] > hma_21[i] and htf_bull:
                desired_signal = REDUCED_SIZE
            elif hma_8[i] < hma_21[i] and htf_bear:
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought OR trend intact
                if crsi[i] < 70 or (hma_8[i] > hma_21[i] and htf_bull):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold OR trend intact
                if crsi[i] > 30 or (hma_8[i] < hma_21[i] and htf_bear):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI overbought AND HTF reverses
            if crsi[i] > 75 and htf_bear:
                desired_signal = 0.0
            # Exit long if HMA crossover reverses in trend mode
            if is_trend and hma_8[i] < hma_21[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI oversold AND HTF reverses
            if crsi[i] < 25 and htf_bull:
                desired_signal = 0.0
            # Exit short if HMA crossover reverses in trend mode
            if is_trend and hma_8[i] > hma_21[i]:
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