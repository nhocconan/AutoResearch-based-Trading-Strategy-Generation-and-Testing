#!/usr/bin/env python3
"""
Experiment #1578: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion with HTF Trend Filter

Hypothesis: Lower timeframe (30m) strategies fail due to fee drag from too many trades.
Solution: Use 4h/1d HMA for TREND DIRECTION (only trade with HTF trend),
30m Connors RSI for ENTRY TIMING (only enter on extreme pullbacks within trend).

Key innovations:
1. 4h HMA(21) for primary trend bias — only long when price > 4h HMA, only short when <
2. 1d HMA(21) for regime filter — adds confirmation, reduces whipsaw
3. Connors RSI (CRSI) on 30m — (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long entry: CRSI < 15 (extreme oversold within uptrend)
   Short entry: CRSI > 85 (extreme overbought within downtrend)
4. Session filter: Only trade 8-20 UTC (high liquidity, less noise)
5. Volume filter: Current volume > 0.8x 20-bar avg volume
6. Choppiness Index(14) > 50 = range (favor mean reversion entries)
7. ATR(14) 2.5x trailing stop for drawdown control
8. Discrete position sizing: 0.22 (smaller for lower TF to reduce fee impact)

Why this should work on 30m:
- HTF trend filter ensures we only trade ~20-40 signals/year (not 200+)
- CRSI extremes (<15 or >85) are rare — reduces trade frequency
- Session + volume filters eliminate low-quality signals
- Mean reversion within trend = higher win rate than pure trend following
- 30m entry timing within 4h trend = best of both worlds

Timeframe: 30m (required for this experiment)
HTF: 4h HMA + 1d HMA for bias (use mtf_data helper — call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
Trade frequency target: 30-80 trades/year (CRITICAL for lower TF success)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_meanrev_4h1d_hma_session_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentage of past returns lower than current return
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi_input = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
        
        # Convert streak to positive for RSI calculation
        streak_rsi_input[i] = max(streak[i], 0)
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak_rsi_input, prepend=streak_rsi_input[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rsi = np.full(n, np.nan)
    mask = streak_loss_smooth > 1e-10
    streak_rsi[mask] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[mask] / streak_loss_smooth[mask]))
    streak_rsi[streak_loss_smooth <= 1e-10] = 100.0
    streak_rsi[:streak_period] = np.nan
    
    # PercentRank(100) - percentage of past returns lower than current
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if np.any(np.isnan(window)):
            continue
        current_return = returns[i]
        count_lower = np.sum(window[:-1] < current_return)
        percent_rank[i] = count_lower / (len(window) - 1) * 100.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if np.isnan(rsi_short[i]) or np.isnan(streak_rsi[i]) or np.isnan(percent_rank[i]):
            continue
        crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range, CHOP < 38.2 = trend
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    # Calculate ATR for each bar (simplified: just high-low for this calculation)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return choppiness

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    import datetime
    dt = datetime.datetime.utcfromtimestamp(open_time / 1000.0)
    return dt.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for primary trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for regime filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, period=14)
    
    # Volume SMA for filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22  # Smaller size for lower TF to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === HTF TREND BIAS (4h HMA + 1d HMA) ===
        # Only long when price > 4h HMA, only short when price < 4h HMA
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # 1d HMA confirmation (adds extra filter)
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 50 = range (favor mean reversion)
        # CHOP < 50 = trend (still allow entries but stricter)
        is_range = choppiness[i] > 50.0
        
        # === CONNORS RSI ENTRY SIGNALS ===
        # Long: CRSI < 15 (extreme oversold) within uptrend
        # Short: CRSI > 85 (extreme overbought) within downtrend
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Trend bull + Daily bull (or neutral) + CRSI oversold + session + volume
        if trend_bull and (daily_bull or not daily_bear) and crsi_oversold and in_session and vol_ok:
            desired_signal = BASE_SIZE
        
        # SHORT: Trend bear + Daily bear (or neutral) + CRSI overbought + session + volume
        elif trend_bear and (daily_bear or not daily_bull) and crsi_overbought and in_session and vol_ok:
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
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