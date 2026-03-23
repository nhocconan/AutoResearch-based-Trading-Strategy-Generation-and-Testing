#!/usr/bin/env python3
"""
Experiment #1178: 30m Primary + 4h/1d HTF — Regime-Adaptive CRSI + HMA Trend + Session Filter

Hypothesis: After 861+ failed experiments, clear pattern emerges for lower TF (30m):
- 30m alone = too many trades → fee drag kills profit (see #1168, #1175 = 0 trades from over-filtering)
- Solution: Use 1d/4h for SIGNAL DIRECTION, 30m only for ENTRY TIMING
- Choppiness Index regime filter: CHOP < 45 = trend (follow HTF), CHOP > 55 = range (mean revert)
- Connors RSI for precise entry timing within HTF trend (CRSI < 20 long, > 80 short)
- Session filter (8-20 UTC) avoids low-liquidity Asian session whipsaws
- Volume confirmation (0.8x avg) ensures real moves
- Position size 0.20 (smaller than 4h strategies) to control 30m volatility
- Target: 40-80 trades/year, Sharpe > 0.612 (beat current best)

Why this should work where #1168, #1175 failed:
- Those had TOO MANY filters = 0 trades
- This uses LOOSER CRSI thresholds (20/80 not 10/90)
- Session filter is binary (in/out), not multiple overlapping conditions
- Volume threshold is 0.8x (not 1.5x) to allow more valid entries
- 1d HMA + 4h HMA confluence = strong trend filter without being too restrictive

Timeframe: 30m (primary)
HTF: 4h + 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.20 (discrete: 0.0, ±0.20)
Stoploss: 2.0x ATR trailing (tighter for 30m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_crsi_hma_4h1d_session_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signals.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 20 = oversold (long opportunity)
    CRSI > 80 = overbought (short opportunity)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close[:rsi_period] = np.nan
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        pos_streaks = np.sum(streak_vals > 0)
        if streak_period > 0:
            streak_rsi[i] = 100.0 * pos_streaks / streak_period
        else:
            streak_rsi[i] = 50.0
    streak_rsi[:streak_period] = np.nan
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * rank / (rank_period - 1)
    percent_rank[:rank_period] = np.nan
    
    # Combine
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — identifies ranging vs trending markets.
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_val = highest_high - lowest_low
        
        if range_val > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / range_val) / np.log10(period)
    
    return chop

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

def calculate_session_filter(open_time):
    """
    Session filter — only trade during high-liquidity hours (8-20 UTC).
    Returns boolean array: True = in session, False = out of session.
    """
    n = len(open_time)
    in_session = np.zeros(n, dtype=bool)
    
    for i in range(n):
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // 3600000) % 24
        if 8 <= hour_utc < 20:
            in_session[i] = True
    
    return in_session

def calculate_volume_confirmation(volume, period=20, threshold=0.8):
    """
    Volume confirmation — current volume > threshold * average volume.
    Lower threshold (0.8) to allow more valid entries vs 1.5.
    """
    n = len(volume)
    confirmed = np.zeros(n, dtype=bool)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period - 1, n):
        if vol_avg[i] > 1e-10 and volume[i] > threshold * vol_avg[i]:
            confirmed[i] = True
    
    return confirmed

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
    
    # Calculate and align HTF HMA for macro trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    session_active = calculate_session_filter(open_time)
    vol_confirmed = calculate_volume_confirmation(volume, period=20, threshold=0.8)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        inter_bull = close[i] > hma_4h_aligned[i]
        inter_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_trending = chop[i] < 45.0  # Trending regime
        is_ranging = chop[i] > 55.0   # Ranging regime
        
        # === CRSI ENTRY SIGNALS ===
        crsi_oversold = crsi[i] < 20.0  # Long opportunity
        crsi_overbought = crsi[i] > 80.0  # Short opportunity
        
        # === SESSION & VOLUME FILTERS ===
        in_session = session_active[i]
        volume_ok = vol_confirmed[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + intermediate bull + (trending OR ranging) + CRSI oversold + session + volume
        if macro_bull and inter_bull and crsi_oversold and in_session and volume_ok:
            # In trending regime: follow HTF trend on pullback
            # In ranging regime: mean revert at lower bound
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + intermediate bear + (trending OR ranging) + CRSI overbought + session + volume
        elif macro_bear and inter_bear and crsi_overbought and in_session and volume_ok:
            desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
        # === INTERMEDIATE TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and inter_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and inter_bull:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x for 30m) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if macro_bull and inter_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if macro_bear and inter_bear:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
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