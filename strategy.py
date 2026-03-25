#!/usr/bin/env python3
"""
Experiment #1309: 15m Primary + 1h/1d HTF — CRSI Mean Reversion with Trend Filter

Hypothesis: 15m strategies have failed due to either too many trades (fee drag) or too few (0 trades).
This strategy uses Connors RSI (CRSI) for high-probability mean reversion entries within a 
higher timeframe trend. Key innovations:

1. CRSI (Connors RSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Entry when CRSI < 15 (oversold) in uptrend, or CRSI > 85 (overbought) in downtrend
   - 75% win rate documented in literature for this setup

2. Dual HTF trend filter: 1d HMA(21) for major bias, 1h HMA(21) for intermediate trend
   - Only long when price > 1d_HMA AND 1h_HMA rising
   - Only short when price < 1d_HMA AND 1h_HMA falling

3. Session filter: Only enter 00-12 UTC (London/NY overlap = highest volume)
   - Reduces false signals during low-volume Asian session

4. Volume confirmation: Current volume > SMA(volume, 20)
   - Ensures institutional participation

5. Conservative sizing: 0.15-0.20 for 15m (higher frequency = smaller size)
   - Target: 40-100 trades/year (fee-friendly for 15m)

6. ATR(14) 2.5x trailing stop for risk management

Why this should work on 15m:
- HTF trend filter reduces trade frequency to HTF levels (40-100/yr)
- CRSI catches pullbacks within trend (high win rate)
- Session filter avoids low-volume whipsaws
- Discrete sizing minimizes fee churn

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_mean_reversion_htf_trend_1h1d_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days (streak length)
    PercentRank: Percentage of prior 100 closes that are below current close
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI of Streak Length (2-period RSI on streak values)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute values for RSI calculation
    streak_abs = np.abs(streak)
    # For RSI_Streak, we use 2-period RSI on streak direction
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(2, n):
        up_sum = 0.0
        down_sum = 0.0
        for j in range(i-1, i+1):
            if streak[j] > streak[j-1]:
                up_sum += streak[j] - streak[j-1]
            elif streak[j] < streak[j-1]:
                down_sum += streak[j-1] - streak[j]
        if down_sum == 0:
            streak_rsi[i] = 100.0
        else:
            rs = up_sum / down_sum
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        count_below = np.sum(close[i-100:i] < close[i])
        percent_rank[i] = (count_below / 100.0) * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def calculate_sma(series, period):
    """Simple Moving Average"""
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
    return sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # 15m HMA for local trend confirmation
    hma_15m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period (need 100 bars for CRSI + HTF alignment)
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 0 and utc_hour < 12)
        
        # === TREND DIRECTION (HTF filters) ===
        # 1d HMA bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1h HMA slope (compare to 5 bars ago for stability on 1h-aligned data)
        hma_1h_slope = 0.0
        if i >= 5 and not np.isnan(hma_1h_aligned[i-5]):
            hma_1h_slope = hma_1h_aligned[i] - hma_1h_aligned[i-5]
        
        # 15m price vs 15m HMA for local confirmation
        price_above_15m = close[i] > hma_15m[i]
        price_below_15m = close[i] < hma_15m[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = volume[i] > vol_sma_20[i]
        
        # === CRSI MEAN REVERSION SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Strong oversold
        crsi_overbought = crsi[i] > 85.0  # Strong overbought
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 1h rising + CRSI oversold + session + volume
        if price_above_1d and hma_1h_slope > 0 and price_above_15m:
            if crsi_oversold and in_session and volume_confirm:
                if crsi[i] < 10.0:
                    desired_signal = SIZE_STRONG  # Very oversold
                else:
                    desired_signal = SIZE_BASE  # Moderately oversold
        
        # SHORT: 1d bearish + 1h falling + CRSI overbought + session + volume
        elif price_below_1d and hma_1h_slope < 0 and price_below_15m:
            if crsi_overbought and in_session and volume_confirm:
                if crsi[i] > 90.0:
                    desired_signal = -SIZE_STRONG  # Very overbought
                else:
                    desired_signal = -SIZE_BASE  # Moderately overbought
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
        
        signals[i] = final_signal
    
    return signals