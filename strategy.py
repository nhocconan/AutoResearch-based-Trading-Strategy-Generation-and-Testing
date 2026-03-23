#!/usr/bin/env python3
"""
Experiment #1058: 30m Primary + 4h/1d HTF — Relaxed Regime + Connors RSI Strategy

Hypothesis: After 766+ failed experiments, the critical insight is that 30m strategies
fail due to TOO STRICT entry conditions (0 trades = auto-reject). The winning approach:

1. USE HTF FOR DIRECTION ONLY (4h HMA21, 1d HMA21)
   - 4h HMA = intermediate trend bias
   - 1d HMA = macro trend bias
   - Both must agree for full position size

2. RELAXED ENTRY THRESHOLDS (learned from 0-trade failures):
   - Connors RSI < 25 for long (not <10)
   - Connors RSI > 75 for short (not >90)
   - Volume > 0.5x average (not 0.8x)
   - Choppiness > 50 = range (not >61.8)

3. CONNORS RSI (CRSI) COMPONENTS:
   - RSI(3): short-term momentum
   - RSI_Streak(2): consecutive up/down days
   - PercentRank(100): where price sits in recent range
   - CRSI = (RSI3 + RSI_Streak + PercentRank) / 3

4. REGIME-SWITCHING:
   - CHOP > 55: Range mode → mean revert at CRSI extremes
   - CHOP < 45: Trend mode → follow 4h HMA direction on pullbacks
   - 45-55: Transition → reduce size or hold existing

5. SESSION FILTER (lenient): 6-22 UTC (not 8-20) to catch more trades

6. POSITION SIZING:
   - Full size (0.30): 4h + 1d HMA agree + regime confirmed
   - Half size (0.15): Only one HTF agrees or transition zone
   - Stoploss: 2.5x ATR trailing

Target: 40-80 trades/year, Sharpe > 0.612, ALL symbols positive
Timeframe: 30m (lower TF = more trades, but use HTF to limit frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_relaxed_regime_4h1d_hma_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Long: CRSI < 25 (oversold)
    Short: CRSI > 75 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < pr_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_series = pd.Series(np.concatenate([[0], gain]))
    loss_series = pd.Series(np.concatenate([[0], loss]))
    
    avg_gain = gain_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi3 = 100 - (100 / (1 + rs))
    rsi3[:rsi_period] = np.nan
    
    # Component 2: RSI of Streak (consecutive up/down)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    
    # Smooth streak
    streak_smooth = pd.Series(streak_abs).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    # Normalize to 0-100 scale (max streak ~10 in crypto)
    rsi_streak = np.clip(50 + streak_sign * streak_smooth * 5, 0, 100)
    rsi_streak[:streak_period] = np.nan
    
    # Component 3: PercentRank (where close sits in last 100 bars)
    percent_rank = np.full(n, np.nan)
    for i in range(pr_period, n):
        window = close[i - pr_period + 1:i + 1]
        rank = np.sum(window[:-1] < close[i])  # count how many bars below current
        percent_rank[i] = rank / (pr_period - 1) * 100
    
    # Combine all three components
    valid = ~np.isnan(rsi3) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid] = (rsi3[valid] + rsi_streak[valid] + percent_rank[valid]) / 3
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market ranging vs trending
    CHOP > 61.8 = ranging (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    Using relaxed thresholds: >55 range, <45 trend
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
    """Average True Range for volatility and stops."""
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

def calculate_volume_ratio(volume, period=20):
    """Current volume vs rolling average volume."""
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    if n < period:
        return vol_ratio
    
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=period, min_periods=period).mean().values
    
    vol_ratio = volume / vol_avg
    vol_ratio[:period] = np.nan
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA21 for trend filters
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    FULL_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Ranging market (mean reversion)
        is_trend = chop[i] < 45.0  # Trending market (trend following)
        is_transition = not is_range and not is_trend  # 45-55 zone
        
        # === HTF TREND FILTERS ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Both HTF agree = strong signal
        htf_strong_bull = trend_4h_bull and trend_1d_bull
        htf_strong_bear = trend_4h_bear and trend_1d_bear
        htf_mixed = not htf_strong_bull and not htf_strong_bear
        
        # === SESSION FILTER (lenient: 6-22 UTC) ===
        # 30m bars: 48 per day, index % 48 gives hour
        bar_of_day = i % 48
        hour_utc = bar_of_day * 0.5  # 30m bars
        in_session = 6 <= hour_utc <= 22
        
        # === VOLUME FILTER (relaxed: >0.5x avg) ===
        volume_ok = vol_ratio[i] > 0.5
        
        desired_signal = 0.0
        signal_strength = 0  # 0=none, 1=half, 2=full
        
        # === RANGE MODE: MEAN REVERSION on CRSI extremes ===
        if is_range:
            # Long: CRSI oversold + volume + session + HTF not strongly bearish
            if crsi[i] < 25 and volume_ok and in_session and not trend_1d_bear:
                signal_strength = 2 if htf_strong_bull else 1
            # Short: CRSI overbought + volume + session + HTF not strongly bullish
            elif crsi[i] > 75 and volume_ok and in_session and not trend_1d_bull:
                signal_strength = 2 if htf_strong_bear else 1
        
        # === TREND MODE: FOLLOW HTF DIRECTION on pullbacks ===
        elif is_trend:
            # Long in uptrend: pullback (CRSI < 40) + HTF bullish
            if crsi[i] < 40 and volume_ok and htf_strong_bull:
                signal_strength = 2
            elif crsi[i] < 45 and volume_ok and trend_4h_bull:
                signal_strength = 1
            # Short in downtrend: pullback (CRSI > 60) + HTF bearish
            elif crsi[i] > 60 and volume_ok and htf_strong_bear:
                signal_strength = 2
            elif crsi[i] > 55 and volume_ok and trend_4h_bear:
                signal_strength = 1
        
        # === TRANSITION ZONE: Reduce size, hold existing ===
        elif is_transition:
            # Only enter if very strong CRSI extreme
            if crsi[i] < 15 and volume_ok and not trend_1d_bear:
                signal_strength = 1
            elif crsi[i] > 85 and volume_ok and not trend_1d_bull:
                signal_strength = 1
        
        # Convert signal strength to position size
        if signal_strength == 2:
            desired_signal = FULL_SIZE if htf_strong_bull else -FULL_SIZE if htf_strong_bear else 0
        elif signal_strength == 1:
            # Determine direction from CRSI
            if crsi[i] < 40:
                desired_signal = HALF_SIZE
            elif crsi[i] > 60:
                desired_signal = -HALF_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and HTF not bearish
                if crsi[i] < 70 and not trend_1d_bear:
                    desired_signal = HALF_SIZE if htf_mixed else FULL_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold and HTF not bullish
                if crsi[i] > 30 and not trend_1d_bull:
                    desired_signal = -HALF_SIZE if htf_mixed else -FULL_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if CRSI very overbought or HTF turns bearish
            if crsi[i] > 80 or (trend_1d_bear and crsi[i] > 50):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if CRSI very oversold or HTF turns bullish
            if crsi[i] < 20 or (trend_1d_bull and crsi[i] < 50):
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