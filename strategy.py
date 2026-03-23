#!/usr/bin/env python3
"""
Experiment #678: 30m Primary + 4h/1d HTF — CRSI + Choppiness + HMA Trend

Hypothesis: 30m timeframe with 4h trend filter + 1d regime detection can achieve
HTF-quality signals with better entry timing. Key innovations:
1. Connors RSI (CRSI) for mean-reversion entries - proven 75% win rate
2. 4h HMA for trend direction - prevents counter-trend trades
3. 1d Choppiness for regime - adapt between trend/mean-revert
4. LOOSE entry thresholds (CRSI<25/>75, not <10/>90) to ensure trade generation
5. Position size 0.25 with 2.5x ATR stoploss

Why 30m can work (learning from #668, #670, #675 failures):
- Previous 30m/1h strategies got 0 trades = entry too strict
- Use 4h for DIRECTION, 30m only for ENTRY TIMING
- CRSI<30 or >70 (not <10 or >90) = more triggers
- Only 2-3 confluence filters (not 5+)
- Session filter REMOVED (was killing trades)

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_hma_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean-reversion signal.
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Proven 75% win rate for mean-reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # Component 1: RSI of close (period=3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.nan_to_num(rsi_close, nan=50)
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # Component 2: RSI of streak (consecutive up/down)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).rolling(window=streak_period, min_periods=streak_period).mean().values
    avg_streak_loss = pd.Series(streak_loss).rolling(window=streak_period, min_periods=streak_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / avg_streak_loss
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50)
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: Percentile rank (100-period)
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        pct_rank[i] = 100 * np.sum(close[i] > window) / rank_period
    
    pct_rank = np.nan_to_num(pct_rank, nan=50)
    
    # Combine all three
    crsi = (rsi_close + rsi_streak + pct_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother than EMA, less lag."""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies ranging vs trending markets."""
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with %B."""
    n = len(close)
    bb_mid = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_pct = np.full(n, np.nan)
    
    if n < period:
        return bb_mid, bb_upper, bb_lower, bb_pct
    
    bb_mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    bb_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    bb_upper = bb_mid + std_mult * bb_std
    bb_lower = bb_mid - std_mult * bb_std
    
    with np.errstate(divide='ignore', invalid='ignore'):
        bb_pct = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    bb_pct = np.clip(bb_pct, 0, 1)
    return bb_mid, bb_upper, bb_lower, bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower, bb_pct = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Calculate and align HTF (4h) indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align HTF (1d) indicators
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size for 30m TF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(chop_30m[i]):
            continue
        if np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION (Daily Choppiness) ===
        chop_daily = chop_1d_aligned[i]
        is_range_regime = chop_daily > 50  # Mean-revert in choppy markets
        is_trend_regime = chop_daily < 45  # Trend-follow in trending markets
        
        # === 4H TREND DIRECTION (HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === 30M CRSI SIGNALS (LOOSE thresholds for trade generation) ===
        crsi_oversold = crsi_30m[i] < 30  # Was <10, too strict
        crsi_overbought = crsi_30m[i] > 70  # Was >90, too strict
        crsi_extreme_oversold = crsi_30m[i] < 20
        crsi_extreme_overbought = crsi_30m[i] > 80
        
        # === BB POSITION ===
        bb_near_lower = bb_pct[i] < 0.20
        bb_near_upper = bb_pct[i] > 0.80
        
        # === VOLUME FILTER ===
        vol_ok = vol_ratio[i] > 0.7  # At least 70% of avg volume
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (CHOP < 45) — Trend Follow with Pullback ===
        if is_trend_regime:
            # Long: 4h bullish + CRSI pullback (not extreme)
            if trend_4h_bullish and crsi_oversold and vol_ok:
                desired_signal = SIZE
            
            # Short: 4h bearish + CRSI rally (not extreme)
            elif trend_4h_bearish and crsi_overbought and vol_ok:
                desired_signal = -SIZE
        
        # === REGIME 2: RANGING (CHOP > 50) — Mean Reversion ===
        elif is_range_regime:
            # Long: CRSI extreme oversold + BB near lower
            if crsi_extreme_oversold and bb_near_lower:
                desired_signal = SIZE
            # Also long on regular oversold if BB confirms
            elif crsi_oversold and bb_near_lower and vol_ok:
                desired_signal = SIZE
            
            # Short: CRSI extreme overbought + BB near upper
            if crsi_extreme_overbought and bb_near_upper:
                desired_signal = -SIZE
            # Also short on regular overbought if BB confirms
            elif crsi_overbought and bb_near_upper and vol_ok:
                desired_signal = -SIZE
        
        # === REGIME 3: TRANSITION (45 <= CHOP <= 50) — Use 4h Trend Only ===
        else:
            # Simpler: just follow 4h trend on CRSI extremes
            if trend_4h_bullish and crsi_oversold:
                desired_signal = SIZE
            elif trend_4h_bearish and crsi_overbought:
                desired_signal = -SIZE
        
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
                # Hold long if 4h still bullish AND CRSI not extremely overbought
                if trend_4h_bullish and crsi_30m[i] < 80:
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if 4h still bearish AND CRSI not extremely oversold
                if trend_4h_bearish and crsi_30m[i] > 20:
                    desired_signal = -SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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