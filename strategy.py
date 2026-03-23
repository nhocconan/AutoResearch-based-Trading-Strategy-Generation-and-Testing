#!/usr/bin/env python3
"""
Experiment #677: 1d Primary + 4h HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Daily timeframe with 4h HTF confirmation provides optimal balance of
signal quality and trade frequency. Key innovations:
1. Connors RSI (CRSI) — proven 75% win rate for mean-reversion entries
2. Choppiness Index regime — switch between trend-follow and mean-revert
3. 4h HMA for entry timing confirmation (not too slow like 1w)
4. LOOSE CRSI thresholds (15/85 not 10/90) to ensure trade generation
5. SMA200 filter for macro bias (long only above, short only below)
6. Position size 0.25-0.30 with 2.5x ATR trailing stop

Why this should work where #673 failed:
- CRSI is more sensitive than regular RSI for entry timing
- 4h HTF is more responsive than 1w for crypto volatility
- Simpler logic = fewer conditions that can conflict = more trades
- SMA200 filter prevents counter-trend trades in strong moves

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_hma_4h_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — combines 3 components for mean-reversion signals.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Long: CRSI < 10-15 (oversold)
    Short: CRSI > 85-90 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI(3) on close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_pad = np.concatenate([[0], gain])
    loss_pad = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain_pad).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss_pad).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # Component 2: RSI on streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).rolling(window=streak_period, min_periods=streak_period).mean().values
    avg_streak_loss = pd.Series(streak_loss).rolling(window=streak_period, min_periods=streak_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank of daily returns over 100 days
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    percent_rank = np.full(n, np.nan)
    
    for i in range(rank_period, n):
        window_returns = returns[i - rank_period + 1:i + 1]
        current_return = returns[i]
        rank = np.sum(window_returns < current_return)
        percent_rank[i] = rank / rank_period * 100
    
    percent_rank[:rank_period] = 50  # neutral before warmup
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — identifies ranging vs trending markets.
    CHOP > 61.8 = choppy/ranging (mean-revert)
    CHOP < 38.2 = trending (trend-follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr1 = high[j] - low[j]
                tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
                tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
                atr_sum += max(tr1, tr2, tr3)
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate primary (1d) indicators
    crsi_1d = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma200_1d = calculate_sma(close, period=200)
    
    # Calculate and align HTF (4h) indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # RSI on 4h for momentum confirmation
    def calculate_rsi(close_arr, period=14):
        n = len(close_arr)
        rsi = np.full(n, np.nan)
        if n < period + 1:
            return rsi
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain_pad = np.concatenate([[0], gain])
        loss_pad = np.concatenate([[0], loss])
        avg_gain = pd.Series(gain_pad).rolling(window=period, min_periods=period).mean().values
        avg_loss = pd.Series(loss_pad).rolling(window=period, min_periods=period).mean().values
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / (avg_loss + 1e-10)
            rsi_raw = 100 - (100 / (1 + rs))
            rsi[period:] = rsi_raw[period - 1:]
        return np.clip(rsi, 0, 100)
    
    rsi_4h_raw = calculate_rsi(df_4h['close'].values, period=14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after warmup (200 for SMA + 50 for other indicators)
        # Skip if indicators not ready
        if np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]):
            continue
        if np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(sma200_1d[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(rsi_4h_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_1d[i]
        is_range_regime = chop_value > 55  # Mean-revert in choppy markets
        is_trend_regime = chop_value < 45  # Trend-follow in trending markets
        
        # === MACRO BIAS (SMA200) ===
        above_sma200 = close[i] > sma200_1d[i]
        below_sma200 = close[i] < sma200_1d[i]
        
        # === 4H HTF CONFIRMATION ===
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        rsi_4h_neutral = 40 <= rsi_4h_aligned[i] <= 60
        
        # === CRSI SIGNALS (LOOSE thresholds for trade generation) ===
        crsi_oversold = crsi_1d[i] < 20  # Looser than 10 to get more trades
        crsi_overbought = crsi_1d[i] > 80  # Looser than 90
        crsi_extreme_oversold = crsi_1d[i] < 15
        crsi_extreme_overbought = crsi_1d[i] > 85
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (CHOP < 45) — Trend Follow with Pullback ===
        if is_trend_regime:
            # Long: Above SMA200 + 4h HMA bullish + CRSI pullback
            if above_sma200 and hma_4h_bullish:
                if crsi_oversold or (crsi_1d[i] < 40 and rsi_4h_neutral):
                    desired_signal = SIZE_LONG
            
            # Short: Below SMA200 + 4h HMA bearish + CRSI rally
            elif below_sma200 and hma_4h_bearish:
                if crsi_overbought or (crsi_1d[i] > 60 and rsi_4h_neutral):
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 2: RANGING (CHOP > 55) — Mean Reversion ===
        elif is_range_regime:
            # Long: CRSI extreme oversold + above or near SMA200
            if crsi_extreme_oversold:
                if above_sma200 or (close[i] > sma200_1d[i] * 0.98):
                    desired_signal = SIZE_LONG
            # Short: CRSI extreme overbought + below or near SMA200
            elif crsi_extreme_overbought:
                if below_sma200 or (close[i] < sma200_1d[i] * 1.02):
                    desired_signal = -SIZE_SHORT
        
        # === REGIME 3: TRANSITION (45 <= CHOP <= 55) — Mixed ===
        else:
            # Use 4h HMA direction with CRSI filter
            if hma_4h_bullish and crsi_1d[i] < 50:
                desired_signal = SIZE_LONG * 0.5  # Half size in transition
            elif hma_4h_bearish and crsi_1d[i] > 50:
                desired_signal = -SIZE_SHORT * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h HMA still bullish AND CRSI not extremely overbought
                if hma_4h_bullish and crsi_1d[i] < 85:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if 4h HMA still bearish AND CRSI not extremely oversold
                if hma_4h_bearish and crsi_1d[i] > 15:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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