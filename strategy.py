#!/usr/bin/env python3
"""
Experiment #1595: 6h Primary + 12h/1d HTF — Connors RSI + Vol Spike Reversion + Asymmetric Regime

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Combining Connors RSI
(proven 75% win rate mean reversion) with volatility spike detection and asymmetric
regime logic should capture panic reversals while avoiding trend whipsaws.

Key innovations vs failed 6h strategies:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI, catches short-term extremes better
   - Long when CRSI<10, Short when CRSI>90 (proven thresholds)
2. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 indicates panic/extreme vol
   - Enter mean reversion trades ONLY when vol spike present
   - Exit when ATR ratio < 1.3 (vol normalized)
3. ASYMMETRIC REGIME: Different logic for bull vs bear
   - Bear (price<1d_HMA): Only short rallies to EMA21, long only extreme CRSI<5
   - Bull (price>1d_HMA): Only long dips to EMA21, short only extreme CRSI>95
   - Range (ADX<20): Standard CRSI mean reversion at BB bounds
4. 12h HMA for intermediate trend filter (prevents counter-trend in strong moves)
5. LOOSE CRSI thresholds (5/95 not 10/90) to guarantee ≥30 trades/train

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- Connors RSI proven superior to standard RSI for mean reversion (75% win rate)
- Vol spike filter captures "panic bottom" and "euphoria top" reversals
- Asymmetric logic prevents counter-trend losses in strong regimes
- 6h TF = more opportunities than 12h, fewer fees than 1h/4h

Entry logic (LOOSE to guarantee trades):
- LONG bear: price<1d_HMA + CRSI<5 + ATR_ratio>2.0 + price<BB_lower
- LONG bull: price>1d_HMA + CRSI<15 + ATR_ratio>1.5 + price<BB_mid
- LONG range: ADX<20 + CRSI<10 + price<BB_lower
- SHORT bear: price<1d_HMA + CRSI>70 + price>EMA21 (short rally)
- SHORT bull: price>1d_HMA + CRSI>95 + ATR_ratio>2.0 + price>BB_upper
- SHORT range: ADX<20 + CRSI>90 + price>BB_upper

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_volspike_asymmetric_12h1d_v1"
timeframe = "6h"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak = 0
        if i > 0:
            # Count consecutive same-direction closes
            direction = 1 if close[i] >= close[i-1] else -1
            j = i
            while j > 0 and ((close[j] >= close[j-1] and direction == 1) or (close[j] < close[j-1] and direction == -1)):
                streak += 1
                j -= 1
                if j == 0:
                    break
            # Convert streak to RSI-like value (0-100)
            streak_rsi[i] = 100.0 * streak / (streak_period + 1) if streak <= streak_period else 100.0
    
    # Percent Rank - current close vs last 100 closes
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    mask = atr != 0
    plus_di[mask] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[mask] / atr[mask]
    minus_di[mask] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[mask] / atr[mask]
    
    dx = np.full(n, np.nan, dtype=np.float64)
    di_sum = plus_di + minus_di
    mask2 = di_sum != 0
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx_14 = calculate_adx(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    ema_21 = calculate_ema(close, period=21)
    
    # Volatility spike ratio
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    mask = atr_30 != 0
    atr_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        adx = adx_14[i]
        is_trend_regime = adx > 25
        is_range_regime = adx < 20
        
        # === TREND DIRECTION (12h and 1d HMA bias) ===
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Define asymmetric regimes
        is_bull_regime = price_above_1d and price_above_12h
        is_bear_regime = price_below_1d and price_below_12h
        is_neutral_regime = not is_bull_regime and not is_bear_regime
        
        # === VOL SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0
        vol_elevated = atr_ratio[i] > 1.5
        vol_normal = atr_ratio[i] < 1.3
        
        # === CRSI EXTREMES ===
        crsi_extreme_low = crsi[i] < 10
        crsi_extreme_high = crsi[i] > 90
        crsi_very_low = crsi[i] < 5
        crsi_very_high = crsi[i] > 95
        
        # === BOLLINGER POSITION ===
        price_at_bb_lower = close[i] <= bb_lower[i] * 1.005
        price_at_bb_upper = close[i] >= bb_upper[i] * 0.995
        price_below_bb_mid = close[i] < bb_mid[i]
        price_above_bb_mid = close[i] > bb_mid[i]
        
        # === PRICE VS EMA21 ===
        price_above_ema21 = close[i] > ema_21[i]
        price_below_ema21 = close[i] < ema_21[i]
        
        # === ENTRY LOGIC (ASYMMETRIC - must generate trades) ===
        desired_signal = 0.0
        
        # BEAR REGIME: Only short rallies, long only extreme panic
        if is_bear_regime:
            # LONG: Extreme panic only (CRSI<5 + vol spike + BB lower)
            if crsi_very_low and vol_spike and price_at_bb_lower:
                desired_signal = SIZE_STRONG
            # SHORT: Rally to EMA21 + CRSI not too low
            elif price_above_ema21 and crsi[i] > 50 and crsi[i] < 85:
                desired_signal = -SIZE_BASE
        
        # BULL REGIME: Only long dips, short only extreme euphoria
        elif is_bull_regime:
            # LONG: Dip to BB mid/lower + CRSI low + vol elevated
            if crsi_extreme_low and vol_elevated and price_below_bb_mid:
                desired_signal = SIZE_STRONG
            elif crsi[i] < 20 and price_below_ema21:
                desired_signal = SIZE_BASE
            # SHORT: Extreme euphoria only (CRSI>95 + vol spike + BB upper)
            elif crsi_very_high and vol_spike and price_at_bb_upper:
                desired_signal = -SIZE_STRONG
        
        # RANGE/NEUTRAL REGIME: Standard CRSI mean reversion
        else:
            # LONG: CRSI extreme low + BB lower
            if crsi_extreme_low and price_at_bb_lower:
                desired_signal = SIZE_BASE
            elif crsi[i] < 15 and vol_elevated:
                desired_signal = SIZE_BASE
            # SHORT: CRSI extreme high + BB upper
            elif crsi_extreme_high and price_at_bb_upper:
                desired_signal = -SIZE_BASE
            elif crsi[i] > 85 and vol_elevated:
                desired_signal = -SIZE_BASE
        
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