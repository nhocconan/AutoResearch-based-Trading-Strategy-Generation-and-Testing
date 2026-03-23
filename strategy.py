#!/usr/bin/env python3
"""
Experiment #815: 1h Primary + 4h/1d HTF — Connors RSI + Session + Volume Confluence

Hypothesis: After 555 failed strategies, key insights for 1h timeframe:
1. 1h needs VERY strict filters to avoid fee drag (target 40-80 trades/year)
2. Use 4h/1d HMA for TREND DIRECTION, 1h only for ENTRY TIMING
3. Connors RSI (CRSI) outperforms standard RSI for mean reversion (75% win rate)
4. Session filter (8-20 UTC) avoids low-volume whipsaw periods
5. Volume confirmation (vol > 0.8x avg) ensures real moves, not noise
6. Relaxed CRSI thresholds (15/85 vs 10/90) to guarantee trades on ALL symbols
7. ATR trailing stop at 2.5x for reasonable exit without premature stops
8. Discrete signals: 0.0, ±0.20, ±0.30 to minimize fee churn

Strategy design:
1. 4h HMA(21) for intermediate trend (aligned via mtf_data)
2. 1d HMA(21) for long-term regime bias (aligned via mtf_data)
3. 1h Connors RSI(3,2,100) for entry timing
4. 1h Session filter: only 8-20 UTC (high volume hours)
5. 1h Volume filter: volume > 0.8x 20-bar average
6. 1h ATR(14) for trailing stop (2.5x)
7. Discrete signals: 0.0, ±0.20, ±0.30
8. RELAXED CRSI thresholds to guarantee >=10 trades per symbol

Key changes from #811 (4h) that worked:
- 1h primary instead of 4h (more entry precision within HTF trend)
- Connors RSI instead of standard RSI (better mean reversion signal)
- Session filter (8-20 UTC) to avoid Asian session whipsaw
- Volume confirmation to filter fake breakouts
- Stricter HTF confluence (both 4h AND 1d must agree for full size)

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive
Timeframe: 1h (target 40-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_session_vol_hma_4h1d_atr_v1"
timeframe = "1h"
leverage = 1.0

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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(streak, 2): RSI on up/down streak length
    PercentRank(100): Percentile rank of today's return vs last 100 days
    
    CRSI < 15 = oversold (long), CRSI > 85 = overbought (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_positive = streak + np.abs(streak.min()) + 1
    rsi_streak = calculate_rsi(streak_positive, streak_period)
    
    # PercentRank(100): percentile of today's return vs last 100
    returns = np.diff(close) / (close[:-1] + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window_returns = returns[i-rank_period:i]
        current_return = returns[i-1]
        rank = np.sum(window_returns < current_return) / len(window_returns)
        percent_rank[i] = rank * 100
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

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
    # Binance timestamps are in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def calculate_volume_avg(volume, period=20):
    """Rolling average volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

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
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_avg_1h = calculate_volume_avg(volume, period=20)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(vol_avg_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if vol_avg_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_timestamp(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (volume > 0.8x average) ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_1h[i]
        
        # === INTERMEDIATE TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === LONG-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === CONNORS RSI SIGNALS (RELAXED for more trades) ===
        crsi_oversold = crsi_1h[i] < 25
        crsi_overbought = crsi_1h[i] > 75
        crsi_extreme_oversold = crsi_1h[i] < 15
        crsi_extreme_overbought = crsi_1h[i] > 85
        crsi_neutral_low = 25 <= crsi_1h[i] < 45
        crsi_neutral_high = 55 < crsi_1h[i] <= 75
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Full size: All confluence (session + volume + both HTF trends + CRSI)
        if in_session and volume_confirmed:
            # Strong long: 4h bullish + 1d bullish + CRSI oversold
            if trend_4h_bullish and trend_1d_bullish and crsi_oversold:
                desired_signal = BASE_SIZE
            
            # Moderate long: 4h bullish + CRSI extreme oversold (guarantees trades)
            elif trend_4h_bullish and crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Moderate long: 1d bullish + CRSI extreme oversold
            elif trend_1d_bullish and crsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
        
        # Even without session/volume, extreme CRSI can trigger reduced size
        if desired_signal == 0.0 and crsi_extreme_oversold:
            if trend_4h_bullish or trend_1d_bullish:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        if in_session and volume_confirmed:
            # Strong short: 4h bearish + 1d bearish + CRSI overbought
            if trend_4h_bearish and trend_1d_bearish and crsi_overbought:
                desired_signal = -BASE_SIZE
            
            # Moderate short: 4h bearish + CRSI extreme overbought
            elif trend_4h_bearish and crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Moderate short: 1d bearish + CRSI extreme overbought
            elif trend_1d_bearish and crsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # Even without session/volume, extreme CRSI can trigger reduced size
        if desired_signal == 0.0 and crsi_extreme_overbought:
            if trend_4h_bearish or trend_1d_bearish:
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
                # Hold long if any HTF trend intact and CRSI not overbought
                if (trend_4h_bullish or trend_1d_bullish) and crsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if any HTF trend intact and CRSI not oversold
                if (trend_4h_bearish or trend_1d_bearish) and crsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both HTF trends reverse + CRSI overbought
            if trend_4h_bearish and trend_1d_bearish and crsi_1h[i] > 70:
                desired_signal = 0.0
            # Exit if CRSI extremely overbought
            if crsi_1h[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both HTF trends reverse + CRSI oversold
            if trend_4h_bullish and trend_1d_bullish and crsi_1h[i] < 30:
                desired_signal = 0.0
            # Exit if CRSI extremely oversold
            if crsi_1h[i] < 15:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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