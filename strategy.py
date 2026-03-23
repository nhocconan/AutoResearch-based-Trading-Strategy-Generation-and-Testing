#!/usr/bin/env python3
"""
Experiment #385: 1h Primary + 4h/1d HTF — Choppiness Regime + CRSI + Session Filter

Hypothesis: Lower timeframe (1h) strategies fail due to either (1) too many trades causing fee drag,
or (2) over-filtering causing 0 trades. This strategy balances both by:

1. CHOPPINESS INDEX (CHOP) regime detection: CHOP>55 = range (mean revert), CHOP<45 = trend (follow)
2. 4h HMA for trend bias (simpler, more stable than KAMA)
3. 1d HMA for macro bias filter
4. Connors RSI with RELAXED thresholds (<30/>70, not <10/>90 which rarely trigger)
5. Session filter: ONLY trade 8-20 UTC (highest volume hours, avoids Asian session whipsaw)
6. Volume confirmation: volume > 0.8x 20-bar average
7. ATR trailing stop: 2.5x ATR

Key innovation: Session filter + volume filter reduces trade count to 30-80/year while
relaxed CRSI thresholds ensure we DON'T get 0 trades (the #1 failure mode in #375,#378,#380,#382,#384).

Target: 40-70 trades/year on 1h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL individually).
Must beat current best: mtf_4h_triple_regime_crsi_donchian_1d1w_v1 (Sharpe=0.612)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_crsi_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    More responsive than EMA, less laggy than SMA.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    hull_raw = 2 * wma_half - wma_full
    hma = hull_raw.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Relaxed thresholds for crypto: <30 oversold, >70 overbought
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, period=rsi_period)
    
    # RSI of Streak - consecutive up/down bars
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= streak_period:
            streak_rsi[i] = 100.0
        elif streak[i] <= -streak_period:
            streak_rsi[i] = 0.0
        else:
            streak_rsi[i] = 50.0 + 25.0 * streak[i]
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank - percentile of today's return vs last pr_period bars
    returns = close_s.pct_change()
    percent_rank = np.full(n, 50.0)
    for i in range(pr_period, n):
        window = returns.iloc[i-pr_period:i]
        if len(window) > 0:
            percent_rank[i] = (returns.iloc[i] > window).sum() / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    choppiness = np.full(n, 50.0)
    
    atr_values = calculate_atr(high, low, close, period=1)  # 1-period ATR = True Range
    
    for i in range(period, n):
        atr_sum = np.sum(atr_values[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    choppiness = np.clip(choppiness, 0, 100)
    return choppiness

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 3600)) % 24
    return hours

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
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    # Calculate and align HTF HMA for bias (4h and 1d)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (target 40-70 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(vol_ma_20[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 8) and (utc_hour <= 20)
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_ma_20[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        is_range = chop_value > 55.0  # Range/choppy market
        is_trend = chop_value < 45.0  # Trending market
        # 45-55 = transition zone, use existing bias
        
        # === HTF BIAS (4h HMA + 1d HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias requires both 4h and 1d aligned
        long_bias = price_above_hma_4h and price_above_hma_1d
        short_bias = price_below_hma_4h and price_below_hma_1d
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === LONG SETUP ===
        # Entry trigger: CRSI oversold
        crsi_oversold = crsi[i] < 30.0
        
        # LONG ENTRY: Need session + volume + bias + (range oversold OR trend pullback)
        if in_session and volume_ok:
            if long_bias:
                if is_range and crsi_oversold:
                    # Range mean-reversion long
                    desired_signal = BASE_SIZE
                elif is_trend and crsi_oversold:
                    # Trend pullback long
                    desired_signal = BASE_SIZE
                elif not is_range and not is_trend and crsi_oversold:
                    # Transition zone with oversold
                    desired_signal = BASE_SIZE
        
        # === SHORT SETUP ===
        # Entry trigger: CRSI overbought
        crsi_overbought = crsi[i] > 70.0
        
        # SHORT ENTRY: Need session + volume + bias + (range overbought OR trend rally)
        if in_session and volume_ok:
            if short_bias:
                if is_range and crsi_overbought:
                    # Range mean-reversion short
                    desired_signal = -BASE_SIZE
                elif is_trend and crsi_overbought:
                    # Trend rally short
                    desired_signal = -BASE_SIZE
                elif not is_range and not is_trend and crsi_overbought:
                    # Transition zone with overbought
                    desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 60:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 40:
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_4h:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_4h:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_4h:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_hma_4h:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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