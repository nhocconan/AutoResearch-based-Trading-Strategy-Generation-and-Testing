#!/usr/bin/env python3
"""
Experiment #027: 1d Primary + 1w HTF — Funding Rate Z-Score + Connors RSI Dual Regime

Hypothesis: Building on #019's success (Sharpe=0.368), this strategy focuses on the 
proven BTC/ETH edge: funding rate mean reversion. Research shows funding z-score 
contrarian strategies achieved Sharpe 0.8-1.5 through the 2022 crash.

Key innovations:
1. Funding rate z-score (30-day) as PRIMARY signal - contrarian when extremes
2. Connors RSI for entry timing - more sensitive than regular RSI, catches reversals
3. 1w HMA for major trend bias - only counter-trend when funding extreme (>2.5 sigma)
4. Choppiness Index regime filter - relaxed thresholds (50/60) to ensure trade generation
5. Asymmetric entry: easier to enter WITH trend, harder against trend
6. Conservative sizing: 0.25 base, 0.30 with full HTF alignment

Why this should work:
- Funding rate extremes precede reversals (crowded longs/shorts unwind)
- CRSI<20 / >80 catches short-term exhaustion better than RSI<30/>70
- 1d primary = 20-40 trades/year target (lower fee drag than lower TF)
- Proven edge for BTC/ETH specifically (SOL is bonus)

Entry Logic:
- Funding z < -2.0 + CRSI < 25 = LONG (crowded shorts, oversold)
- Funding z > +2.0 + CRSI > 75 = SHORT (crowded longs, overbought)
- WITH 1w trend: relax funding threshold to 1.5 sigma
- CHOPPY regime (CHOP>50): mean revert at CRSI extremes
- TREND regime (CHOP<60): follow 1w HMA direction

Risk: 2.5x ATR trailing stop, max signal 0.30, discrete levels (0.0, ±0.25, ±0.30)
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_funding_zscore_crsi_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close,3) + RSI(Streak,2) + PercentRank(100)) / 3
    
    RSI(Streak): RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(close, 3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI(Streak, 2) - streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    for i in range(streak_period, n):
        total = avg_streak_gain[i] + avg_streak_loss[i]
        if total < 1e-10:
            rsi_streak[i] = 50.0
        else:
            rsi_streak[i] = 100.0 * avg_streak_gain[i] / total
    
    # PercentRank(100) - percentage of prior closes lower than current
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_lower / rank_period
    
    # Combine into CRSI
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if np.isnan(rsi_close[i]) or np.isnan(rsi_streak[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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

def calculate_funding_zscore(funding_rates, lookback=30):
    """
    Calculate z-score of funding rates over lookback period
    z = (current_funding - mean) / std
    """
    n = len(funding_rates)
    zscore = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = funding_rates[i-lookback:i]
        mean_funding = np.mean(window)
        std_funding = np.std(window)
        if std_funding > 1e-10:
            zscore[i] = (funding_rates[i] - mean_funding) / std_funding
    
    return zscore

def load_funding_data(symbol):
    """Load funding rate data from processed parquet files"""
    try:
        import os
        symbol_base = symbol.replace('USDT', '').lower()
        funding_path = f"data/processed/funding/{symbol_base}.parquet"
        
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            return {
                'timestamp': df_funding['timestamp'].values,
                'funding_rate': df_funding['funding_rate'].values
            }
    except Exception:
        pass
    
    return None

def get_funding_series(funding_data, open_time):
    """Align funding data to price timestamps"""
    if funding_data is None:
        return np.zeros(len(open_time)), np.zeros(len(open_time))
    
    ts_arr = funding_data['timestamp']
    fr_arr = funding_data['funding_rate']
    
    funding_aligned = np.zeros(len(open_time))
    funding_ts = np.zeros(len(open_time))
    
    for i, ts in enumerate(open_time):
        idx = np.searchsorted(ts_arr, ts)
        if idx >= len(ts_arr):
            idx = len(ts_arr) - 1
        if idx < 0:
            idx = 0
        funding_aligned[i] = fr_arr[idx]
        funding_ts[i] = ts_arr[idx]
    
    return funding_aligned, funding_ts

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Load and align funding data
    funding_data = None
    try:
        funding_data = load_funding_data("BTCUSDT")
    except Exception:
        funding_data = None
    
    funding_rates, funding_ts = get_funding_series(funding_data, open_time)
    funding_z = calculate_funding_zscore(funding_rates, lookback=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    FULL_SIZE = 0.30
    MAX_SIZE = 0.30
    
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
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(funding_z[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # Relaxed thresholds to ensure trade generation
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 60.0
        
        # === HTF TREND BIAS (1w) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === FUNDING RATE Z-SCORE (PRIMARY SIGNAL) ===
        funding_extreme_long = funding_z[i] < -2.0  # Crowded shorts → long
        funding_extreme_short = funding_z[i] > 2.0  # Crowded longs → short
        
        # Relax threshold when WITH 1w trend
        funding_mild_long = funding_z[i] < -1.5 and hma_1w_bull
        funding_mild_short = funding_z[i] > 1.5 and hma_1w_bear
        
        # === CRSI ENTRY TIMING ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # LONG entries (funding extreme + CRSI oversold)
        if (funding_extreme_long or funding_mild_long) and crsi_oversold:
            if hma_1w_bull:
                signal_strength = FULL_SIZE  # WITH trend
            else:
                signal_strength = BASE_SIZE  # Counter-trend (need stronger funding signal)
            desired_signal = signal_strength
        
        # SHORT entries (funding extreme + CRSI overbought)
        elif (funding_extreme_short or funding_mild_short) and crsi_overbought:
            if hma_1w_bear:
                signal_strength = FULL_SIZE  # WITH trend
            else:
                signal_strength = BASE_SIZE  # Counter-trend
            desired_signal = -signal_strength
        
        # === CHOPPY REGIME: Pure mean reversion (relaxed funding requirement) ===
        if is_choppy and desired_signal == 0.0:
            # In choppy markets, funding threshold relaxed
            if funding_z[i] < -1.0 and crsi_oversold:
                desired_signal = BASE_SIZE
            elif funding_z[i] > 1.0 and crsi_overbought:
                desired_signal = -BASE_SIZE
        
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
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= FULL_SIZE * 0.85:
            final_signal = FULL_SIZE
        elif desired_signal <= -FULL_SIZE * 0.85:
            final_signal = -FULL_SIZE
        elif desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif abs(desired_signal) >= 0.10:
            final_signal = np.sign(desired_signal) * BASE_SIZE
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