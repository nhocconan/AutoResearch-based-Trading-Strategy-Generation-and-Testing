#!/usr/bin/env python3
"""
Experiment #1168: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Volume + Session

Hypothesis: After 850+ failed experiments, the pattern for lower TF (30m) is clear:
- Use HTF (4h/1d) for SIGNAL DIRECTION, not entry timing
- Use 30m only for precise ENTRY within HTF trend
- Choppiness Index determines regime: CHOP>55=range (mean revert), CHOP<45=trend (follow)
- Connors RSI (RSI3 + RSI_Streak2 + PercentRank100)/3 for entry timing
- Session filter: only trade 6-22 UTC (high liquidity hours)
- Volume filter: only trade when volume > 0.6x 20-period average
- Position size: 0.20 (smaller for lower TF to reduce fee drag)
- Target: 40-80 trades/year (strict enough to avoid fee drag, loose enough to generate trades)

Why this should beat Sharpe=0.612:
- Regime-adaptive: different logic for range vs trend markets
- Connors RSI has 75% win rate in backtests (original research)
- Session filter avoids low-liquidity hours (reduces slippage)
- Volume filter confirms genuine moves (not fake breakouts)
- 4h/1d HTF prevents counter-trend trades in major moves
- Looser thresholds than #1158/#1160/#1165 to ensure >=10 trades per symbol

Timeframe: 30m (primary)
HTF: 4h, 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.20 base (discrete: 0.0, ±0.20)
Stoploss: 2.0x ATR trailing (tighter for 30m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_regime_crsi_4h1d_session_vol_atr_v1"
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

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    """
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signal.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    CRSI < 30 = oversold (long signal)
    CRSI > 70 = overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi3 = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of Streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 1 and close[i-1] > close[i-2] else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 1 and close[i-1] < close[i-2] else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_gain = np.zeros(n)
    streak_loss = np.zeros(n)
    streak_delta = np.diff(streak)
    
    streak_gain[1:] = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss[1:] = np.where(streak_delta < 0, -streak_delta, 0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan)
    mask = streak_loss_smooth > 1e-10
    rs_streak = np.zeros(n)
    rs_streak[mask] = streak_gain_smooth[mask] / streak_loss_smooth[mask]
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak[:streak_period] = np.nan
    
    # Component 3: Percent Rank of returns
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1]
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) >= rank_period // 2:
            current_return = returns[i]
            if not np.isnan(current_return):
                rank = np.sum(valid_window < current_return)
                percent_rank[i] = rank / len(valid_window) * 100.0
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 55 = choppy/range (mean reversion)
    CHOP < 45 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate Choppiness Index
    for i in range(period - 1, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
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
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Volume average for filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or atr[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (6-22 UTC only) ===
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 6 <= hour_utc <= 22
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.6 * vol_avg[i]
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        inter_bull = close[i] > hma_4h_aligned[i]
        inter_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 30.0
        crsi_overbought = crsi[i] > 70.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Range regime: CRSI oversold + session + volume
        if is_range and crsi_oversold and in_session and volume_ok:
            if not macro_bear or inter_bull:
                desired_signal = BASE_SIZE
        
        # Trend regime: Follow HTF trend + CRSI pullback
        elif is_trend and in_session and volume_ok:
            if macro_bull and inter_bull and crsi[i] < 60.0:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Range regime: CRSI overbought + session + volume
        if is_range and crsi_overbought and in_session and volume_ok:
            if not macro_bull or inter_bear:
                desired_signal = -BASE_SIZE
        
        # Trend regime: Follow HTF trend + CRSI pullback
        elif is_trend and in_session and volume_ok:
            if macro_bear and inter_bear and crsi[i] > 40.0:
                desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear and inter_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull and inter_bull:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if (is_range or (is_trend and macro_bull and inter_bull)) and in_session:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if (is_range or (is_trend and macro_bear and inter_bear)) and in_session:
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