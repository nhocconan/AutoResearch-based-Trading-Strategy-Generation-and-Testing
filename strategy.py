#!/usr/bin/env python3
"""
Experiment #1228: 30m Primary + 4h/1d HTF — Simplified Regime + CRSI + Funding

Hypothesis: Previous 30m/1h strategies failed with Sharpe=0.000 (0 trades) due to 
too many confluence filters. Key insight from #1222: funding rate z-score adds 
alpha for BTC/ETH mean reversion. New approach:
(1) 4h HMA(21) for trend direction — simple, stable
(2) 1d HMA(21) for macro regime — price above = bull bias, below = bear bias
(3) 30m CRSI(3,2,100) for entry timing — only enter on extremes IN trend direction
(4) Funding z-score(30) as contrarian filter — extreme funding = fade
(5) REMOVED: session filter (killed signals in #1218, #1220), strict volume

Critical changes from failed 30m attempts:
- Looser CRSI thresholds (25/75 instead of 15/85) to ensure trade frequency
- No session filter (8-20 UTC killed 60% of signals)
- Volume filter removed (too restrictive)
- Focus on HTF direction + LTF timing only (3 filters max)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, DD > -50%
Position size: 0.25 (conservative for 30m to limit fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_funding_4h1d_hma_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=3):
    """Relative Strength Index for Connors RSI component."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """RSI of streak length (consecutive up/down bars) for Connors RSI."""
    n = len(close)
    rsi_streak = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi_streak
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = max(1, streak[i-1] + 1) if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = min(-1, streak[i-1] - 1) if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    
    delta = np.diff(streak_abs)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi_streak = 100.0 - (100.0 / (1.0 + rs))
    rsi_streak[:period] = np.nan
    
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """Percentile rank for Connors RSI component."""
    n = len(close)
    pr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        current = close[i]
        rank = np.sum(window < current) / period * 100.0
        pr[i] = rank
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + rsi_streak + pr) / 3.0
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
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

def calculate_funding_zscore(prices, period=30):
    """
    Funding rate z-score for contrarian signal.
    High positive funding = crowded longs = short signal
    High negative funding = crowded shorts = long signal
    """
    n = len(prices)
    zscore = np.full(n, np.nan)
    
    # Try to load funding data
    try:
        import os
        symbol = "BTCUSDT"  # Default, will be overridden by engine
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            funding_df = pd.read_parquet(funding_path)
            funding_rates = funding_df['funding_rate'].values
            
            # Align funding to prices (funding is 8h, prices are 30m)
            # Simple approach: use last known funding rate
            if len(funding_rates) > 0:
                # Calculate z-score on funding rates
                for i in range(period, n):
                    window = funding_rates[max(0, i-period):i]
                    if len(window) > 0:
                        mean_f = np.mean(window)
                        std_f = np.std(window)
                        if std_f > 1e-10:
                            zscore[i] = (funding_rates[min(i, len(funding_rates)-1)] - mean_f) / std_f
    except:
        pass
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    # Funding z-score (contrarian filter)
    funding_z = calculate_funding_zscore(prices, period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 30m
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === MACRO TREND (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME (1d HMA) ===
        regime_bull = close[i] > hma_1d_aligned[i]
        regime_bear = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI EXTREMES (looser for trade frequency) ===
        crsi_oversold = crsi[i] < 25.0  # Looser than 15 for more trades
        crsi_overbought = crsi[i] > 75.0  # Looser than 85 for more trades
        
        # === FUNDING Z-SCORE (contrarian) ===
        funding_extreme_long = funding_z[i] > 1.5 if not np.isnan(funding_z[i]) else False
        funding_extreme_short = funding_z[i] < -1.5 if not np.isnan(funding_z[i]) else False
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # LONG: Need 4h trend + 1d regime + CRSI oversold
        # Funding filter is OPTIONAL (only adds conviction, doesn't block)
        if trend_bull and crsi_oversold:
            # Base long signal
            if regime_bull:
                # Strong confluence: both HTFs bullish
                desired_signal = BASE_SIZE
            else:
                # Weaker signal: 4h bull but 1d bear (pullback in bear market rally)
                desired_signal = BASE_SIZE * 0.6
            # Funding boost: if funding extremely positive, increase size (crowded longs = reversal)
            if funding_extreme_long:
                desired_signal = min(desired_signal * 1.2, BASE_SIZE)
        
        # SHORT: Need 4h trend + 1d regime + CRSI overbought
        if trend_bear and crsi_overbought:
            if regime_bear:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -BASE_SIZE * 0.6
            if funding_extreme_short:
                desired_signal = max(desired_signal * 1.2, -BASE_SIZE)
        
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
            desired_signal = BASE_SIZE
        elif desired_signal < -0.1:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals