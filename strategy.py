#!/usr/bin/env python3
"""
Experiment #014: 1h CRSI + Donchian with 4h Trend Filter

HYPOTHESIS: 1h CRSI (Connors RSI) combined with Donchian channel breaks
captures mean-reversion entries within 4h trend context. CRSI < 20 signals
oversold (long), CRSI > 80 signals overbought (short). 4h HMA provides trend
bias. Session filter (08-20 UTC) reduces noise from thin overnight trading.
Targeting 60-150 trades over 4 years.

KEY INSIGHT from DB winners: CRSI-based entries consistently achieve
Sharpe 1.3-1.8 on test. The combination of RSI(3) + streak + percent rank
catches reversals better than plain RSI.

TIMEFRAME: 1h primary
HTF: 4h for trend bias
TARGET: 60-150 total trades over 4 years (15-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_donchian_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_crsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    rsi_vals = np.full(n, np.nan, dtype=np.float64)
    
    # RSI(3)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    avg_loss = loss.ewm(span=period_rsi, min_periods=period_rsi, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = (100 - (100 / (1 + rs))).values
    
    # RSI Streak
    rsi_streak = np.full(n, 50.0, dtype=np.float64)
    streak_count = 0
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak_count = max(0, streak_count + 1)
        elif delta.iloc[i] < 0:
            streak_count = min(0, streak_count - 1)
        
        # RSI of streak (simplified: convert streak to oscillator)
        streak_signal = (streak_count / period_streak) * 50 + 50
        streak_signal = np.clip(streak_signal, 0, 100)
        rsi_streak[i] = streak_signal
    
    # Percent Rank (100 period)
    pct_rank = np.full(n, 50.0, dtype=np.float64)
    lookback = period_rank
    for i in range(lookback, n):
        window = close[i - lookback + 1:i + 1]
        rank = (close[i] - np.min(window)) / (np.max(window) - np.min(window) + 1e-10)
        pct_rank[i] = rank * 100
    
    # CRSI = average of three
    crsi = (rsi_3 + rsi_streak + pct_rank) / 3.0
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_4h = get_htf_data(prices, '4h')
    
    # 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate local 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # 1h Donchian 20-period
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    donch_mid = (donch_upper + donch_lower) / 2
    
    # CRSI
    crsi = calculate_crsi(close)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Session filter - precompute hours (08-20 UTC)
    hours = prices.index.hour
    in_session = ((hours >= 8) & (hours <= 20))
    
    signals = np.zeros(n)
    SIZE = 0.20  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    cooldown_bars = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter
        session_ok = in_session[i]
        
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Cooldown
        if cooldown_bars > 0:
            cooldown_bars -= 1
        
        # === TREND (4h HMA) ===
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        bullish_trend = price_above_4h_hma
        bearish_trend = not price_above_4h_hma
        
        # === MOMENTUM (CRSI) ===
        crsi_val = crsi[i] if not np.isnan(crsi[i]) else 50.0
        
        # === DONCHIAN POSITION ===
        donch_width = donch_upper[i] - donch_lower[i]
        price_vs_mid = (close[i] - donch_lower[i]) / (donch_width + 1e-10)
        
        # === VOLUME ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if cooldown_bars == 0:
            # === LONG ENTRY ===
            # CRSI < 20 (oversold) + above 4h HMA trend + not in position
            if crsi_val < 20 and bullish_trend:
                # Price near or below lower band - mean reversion long
                if price_vs_mid < 0.4 or low[i] < donch_lower[i] * 1.002:
                    desired_signal = SIZE
                    cooldown_bars = 4  # Min hold ~4 hours
            
            # === SHORT ENTRY ===
            # CRSI > 80 (overbought) + below 4h HMA trend
            if crsi_val > 80 and bearish_trend:
                # Price near or above upper band - mean reversion short
                if price_vs_mid > 0.6 or high[i] > donch_upper[i] * 0.998:
                    desired_signal = -SIZE
                    cooldown_bars = 4
        
        # === STOPLOSS (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            bars_in_trade += 1
            # Exit: CRSI > 60 (mean reversion complete) OR opposite trend signal
            if crsi_val > 60:
                exit_triggered = True
            # Exit on strong opposing trend
            if bearish_trend and crsi_val > 50:
                exit_triggered = True
            # Time-based exit (min 6 bars, max 48 bars)
            if bars_in_trade >= 48:
                exit_triggered = True
            if bars_in_trade >= 6 and crsi_val > 40:
                exit_triggered = True
        
        if in_position and position_side < 0:
            bars_in_trade += 1
            # Exit: CRSI < 40 (mean reversion complete)
            if crsi_val < 40:
                exit_triggered = True
            # Exit on strong opposing trend
            if bullish_trend and crsi_val < 50:
                exit_triggered = True
            # Time-based exit
            if bars_in_trade >= 48:
                exit_triggered = True
            if bars_in_trade >= 6 and crsi_val < 60:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_in_trade = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals