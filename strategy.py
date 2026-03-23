#!/usr/bin/env python3
"""
Experiment #1048: 30m Primary + 4h/1d HTF — Regime-Adaptive CRSI Strategy

Hypothesis: After 758 failed experiments, the key insight for 30m is:
1. Use 4h/1d HTF for SIGNAL DIRECTION only (not entry timing)
2. Use 30m for ENTRY TRIGGER only (when to pull trigger within HTF trend)
3. Relaxed thresholds are CRITICAL — #1 failure is 0 trades from over-filtering
4. Connors RSI (CRSI) proven 75% win rate in mean reversion
5. Choppiness Index regime filter adapts to market conditions

Strategy Logic:
- 4h HMA21: Macro trend direction (long only when bullish)
- 1d HMA50: Longer-term bias filter (avoid counter-trend)
- CRSI(3,2,100): Entry trigger (oversold <30 / overbought >70)
- Volume: >0.6x 20-bar avg (not too strict)
- Session: 8-20 UTC for entries only (liquidity)
- CHOP(14): Regime detection (range vs trend mode)
- ATR(14) 2.5x: Trailing stoploss

CRITICAL FOR 30m: Relaxed thresholds to ensure 30-80 trades/year
- CRSI: <30 / >70 (not <10 / >90)
- Volume: >0.6x avg (not >1.0x)
- ADX: >15 for trend mode (not >25)
- Session filter on ENTRY only, not exit

Position Size: 0.25 discrete (smaller for lower TF fee sensitivity)
Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_regime_4h1d_hma_session_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 momentum components
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    Proven 75% win rate for mean reversion when CRSI < 10 (long) or > 90 (short)
    We use relaxed <30 / >70 for 30m to ensure sufficient trades
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI of close (period=3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(np.concatenate([[0], gain]))
    loss_series = pd.Series(np.concatenate([[0], loss]))
    
    avg_gain = gain_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_close = 100 - (100 / (1 + rs))
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    streak_gain_series = pd.Series(np.concatenate([[0], streak_gain]))
    streak_loss_series = pd.Series(np.concatenate([[0], streak_loss]))
    
    avg_streak_gain = streak_gain_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = streak_loss_series.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.divide(avg_streak_gain, avg_streak_loss, out=np.zeros_like(avg_streak_gain), where=avg_streak_loss != 0)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Component 3: PercentRank of returns (lookback=100)
    returns = np.diff(close) / close[:-1]
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current_return = returns[i-1] if i > 0 else 0
        percent_rank[i] = np.sum(window < current_return) / rank_period * 100
    
    percent_rank[0] = 50  # neutral for first bar
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market ranging vs trending
    CHOP > 61.8 = ranging (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
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
    
    # Calculate and align 4h HMA21 for macro trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA50 for longer-term bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume moving average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for 30m (fee sensitivity)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Ranging market
        is_trend = chop[i] < 45.0  # Trending market
        # 45-55: transition zone, use trend bias
        
        # === MACRO TREND FILTERS ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === SESSION FILTER (8-20 UTC) for entries only ===
        hour_utc = (open_time[i] // 3600000) % 24
        is_liquid_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER (relaxed) ===
        volume_ok = volume[i] > 0.6 * vol_ma[i]
        
        desired_signal = 0.0
        
        # === RANGE MODE: MEAN REVERSION (CRSI extremes) ===
        if is_range:
            # Long: CRSI oversold + 4h trend bullish + volume + session
            if crsi[i] < 30 and trend_4h_bull and volume_ok and is_liquid_session:
                desired_signal = BASE_SIZE
            # Short: CRSI overbought + 4h trend bearish + volume + session
            elif crsi[i] > 70 and trend_4h_bear and volume_ok and is_liquid_session:
                desired_signal = -BASE_SIZE
            # Weaker signals without session filter (hold existing positions)
            elif crsi[i] < 25 and trend_4h_bull and volume_ok:
                desired_signal = BASE_SIZE * 0.8
            elif crsi[i] > 75 and trend_4h_bear and volume_ok:
                desired_signal = -BASE_SIZE * 0.8
        
        # === TREND MODE: TREND FOLLOWING ===
        elif is_trend:
            # Long: 4h bullish + 1d neutral/bullish + CRSI not overbought
            if trend_4h_bull and not trend_1d_bear and crsi[i] < 60 and volume_ok:
                desired_signal = BASE_SIZE
            # Short: 4h bearish + 1d neutral/bearish + CRSI not oversold
            elif trend_4h_bear and not trend_1d_bull and crsi[i] > 40 and volume_ok:
                desired_signal = -BASE_SIZE
            # Pullback entries in trend
            elif trend_4h_bull and crsi[i] < 40 and volume_ok:
                desired_signal = BASE_SIZE * 0.8
            elif trend_4h_bear and crsi[i] > 60 and volume_ok:
                desired_signal = -BASE_SIZE * 0.8
        
        # === TRANSITION ZONE (45-55 CHOP) ===
        else:
            # Use 4h trend bias with CRSI confirmation
            if trend_4h_bull and crsi[i] < 35 and volume_ok:
                desired_signal = BASE_SIZE * 0.8
            elif trend_4h_bear and crsi[i] > 65 and volume_ok:
                desired_signal = -BASE_SIZE * 0.8
        
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
        
        # === HOLD LOGIC — Maintain position if thesis intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h still bullish or CRSI not overbought
                if trend_4h_bull or crsi[i] < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h still bearish or CRSI not oversold
                if trend_4h_bear or crsi[i] > 35:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h reverses bearish AND CRSI overbought
            if trend_4h_bear and crsi[i] > 70:
                desired_signal = 0.0
            # Exit long if 1d strongly bearish
            if trend_1d_bear and crsi[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h reverses bullish AND CRSI oversold
            if trend_4h_bull and crsi[i] < 30:
                desired_signal = 0.0
            # Exit short if 1d strongly bullish
            if trend_1d_bull and crsi[i] < 40:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE:
                desired_signal = BASE_SIZE
            elif desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE * 0.8
            else:
                desired_signal = BASE_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE * 0.8
            else:
                desired_signal = -BASE_SIZE * 0.5
        
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