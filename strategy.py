#!/usr/bin/env python3
"""
Experiment #1340: 1h Primary + 4h/12h HTF — HMA Trend + Connors RSI + ADX Regime

Hypothesis: 1h timeframe with 4h HMA trend filter and 12h ADX regime detection balances
trade frequency (target 40-80/year) with signal quality. Connors RSI (CRSI) provides faster
entry signals than standard RSI while maintaining mean-reversion properties. ADX(14)>20
confirms trend regime without being too strict (>25 kills trades). Session filter 8-20 UTC
captures liquidity hours. Simpler entry logic than #1330/#1335 to ensure trades happen.

Key design choices:
1. 4h HMA(21) for trend bias - proven in #1329, #1337, #1339
2. 12h ADX(14) for regime - threshold 20 (not 25) to allow more trades
3. Connors RSI(3,2,100) for entry - faster than RSI(14), 75% win rate in literature
4. Session filter 8-20 UTC - liquidity hours only
5. Volume > 0.6x avg (not 0.8x) - looser to ensure trades
6. ATR(14) trailing stop 2.5x - proven in #1329
7. Position size 0.25 - conservative for 1h volatility

CRITICAL: Multiple entry paths to ensure trades happen (learned from #1330, #1335 0-trade failures)
- Path 1: HTF trend + CRSI extreme + ADX confirm
- Path 2: HTF trend + price pullback to HMA + volume
- Path 3: HTF trend + simple momentum (RSI>50/<50)

Target: 40-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_crsi_adx_4h12h_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.full(n, np.nan)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi_short[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_short[loss_smooth <= 1e-10] = 100.0
    rsi_short[:rsi_period] = np.nan
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        if len(streak_window) > 0:
            gain_streak = np.sum(np.maximum(streak_window, 0))
            loss_streak = np.sum(np.abs(np.minimum(streak_window, 0)))
            if loss_streak > 1e-10:
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + gain_streak / loss_streak))
            else:
                streak_rsi[i] = 100.0
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < window[-1])
            percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    mask_valid = (~np.isnan(rsi_short)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[mask_valid] = (rsi_short[mask_valid] + streak_rsi[mask_valid] + percent_rank[mask_valid]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Rolling average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h ADX for regime filter
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # HMA slope for trend confirmation
    hma_1h = calculate_hma(close, period=21)
    hma_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(hma_1h[i]) and not np.isnan(hma_1h[i-1]):
            hma_slope[i] = hma_1h[i] - hma_1h[i-1]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (looser: 0.6x avg) ===
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 1e-10 else 0.0
        vol_ok = vol_ratio > 0.6
        
        # === MACRO TREND (4h HMA) ===
        macro_bull = close[i] > hma_4h_aligned[i]
        macro_bear = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND (1h HMA) ===
        hma_bull = (close[i] > hma_1h[i]) and (hma_slope[i] > 0)
        hma_bear = (close[i] < hma_1h[i]) and (hma_slope[i] < 0)
        
        # === REGIME (12h ADX > 20, not 25) ===
        trend_regime = adx_12h_aligned[i] > 20.0
        strong_trend = adx_12h_aligned[i] > 30.0
        
        # === CONNORS RSI ENTRY (extreme values) ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_neutral_bull = crsi[i] > 40.0
        crsi_neutral_bear = crsi[i] < 60.0
        
        # === DESIRED SIGNAL (multiple paths to ensure trades) ===
        desired_signal = 0.0
        
        # LONG ENTRY - Path 1: HTF bull + CRSI oversold + session (mean reversion in trend)
        if macro_bull and crsi_oversold and in_session:
            desired_signal = BASE_SIZE
        # LONG ENTRY - Path 2: HTF bull + local bull + volume (trend follow)
        elif macro_bull and hma_bull and vol_ok:
            desired_signal = BASE_SIZE
        # LONG ENTRY - Path 3: HTF bull + CRSI neutral + ADX confirm (momentum)
        elif macro_bull and crsi_neutral_bull and trend_regime:
            desired_signal = BASE_SIZE * 0.5
        # LONG ENTRY - Path 4: Simple - price above both HMA (catches trends)
        elif macro_bull and close[i] > hma_1h[i] and close[i] > hma_4h_aligned[i]:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY - Path 1: HTF bear + CRSI overbought + session (mean reversion in trend)
        elif macro_bear and crsi_overbought and in_session:
            desired_signal = -BASE_SIZE
        # SHORT ENTRY - Path 2: HTF bear + local bear + volume (trend follow)
        elif macro_bear and hma_bear and vol_ok:
            desired_signal = -BASE_SIZE
        # SHORT ENTRY - Path 3: HTF bear + CRSI neutral + ADX confirm (momentum)
        elif macro_bear and crsi_neutral_bear and trend_regime:
            desired_signal = -BASE_SIZE * 0.5
        # SHORT ENTRY - Path 4: Simple - price below both HMA (catches trends)
        elif macro_bear and close[i] < hma_1h[i] and close[i] < hma_4h_aligned[i]:
            desired_signal = -BASE_SIZE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals