#!/usr/bin/env python3
"""
Experiment #161: 15m Primary + 1h/4h/1d HTF — HMA Trend + RSI Momentum + Session Filter

Hypothesis: 15m timeframe is UNDEREXPLORED (ZERO experiments in history) and offers
opportunity for higher Sharpe with proper HTF filtering. Key insight from failures:
- #149, #153, #157: 15m strategies got Sharpe=0.000 (ZERO trades) from overly strict conditions
- Lower TF needs HTF direction filter to avoid fee drag from whipsaws
- Session filter (UTC 00-12) captures London/NY overlap volatility

Strategy design:
- 15m HMA(21) for entry timing (faster than EMA, less lag than SMA)
- 1h HMA(34) for intraday trend bias
- 4h HMA(50) for major trend confirmation
- 1d HMA(50) for regime filter (only trade with daily trend)
- RSI(7) for momentum (faster than RSI(14) for 15m entries)
- Session filter: only trade UTC 00-12 (London+NY overlap = 75% of crypto volume)
- ATR ratio filter: avoid extreme volatility entries (ATR7/ATR30 < 2.0)
- Position size: 0.18 (18% - smaller for higher frequency)
- Target: 50-100 trades/year on 15m (strict confluence to avoid fee drag)

Trade generation design (CRITICAL - avoid 0 trades like #149, #153, #157):
- LOOSE RSI thresholds (>40 long, <60 short) to ensure entries
- Multiple entry tiers (full size when all HTF align, partial when fewer align)
- Fallback entries when 15m + 1h align strongly (ignore 4h/1d)
- Session filter relaxes on strong signals (allow 12-16 UTC for 80% size)

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_1h4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA, smoother than SMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - standard Wilder's formula"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio - measures vol expansion vs baseline"""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    n = len(close)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_utc_hour(prices, idx):
    """Extract UTC hour from open_time timestamp"""
    # open_time is in milliseconds since epoch
    ts_ms = prices['open_time'].iloc[idx]
    ts_sec = ts_ms / 1000.0
    utc_hour = (ts_sec % 86400) // 3600
    return int(utc_hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for intraday trend
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=34)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 4h HMA for major trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for regime filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (smaller for 15m frequency)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
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
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC hours) ===
        utc_hour = get_utc_hour(prices, i)
        # Prime session: 00-12 UTC (London open to NY close overlap)
        # Secondary: 12-16 UTC (NY afternoon)
        # Avoid: 16-00 UTC (Asia session = lower volume, more whipsaws)
        prime_session = 0 <= utc_hour < 12
        secondary_session = 12 <= utc_hour < 16
        
        # === HTF BIAS (1h HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === MAJOR TREND (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY FILTER ===
        vol_ok = atr_ratio[i] < 2.0  # Avoid extreme vol spikes
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI MOMENTUM (LOOSE thresholds for trade generation) ===
        rsi_7_ok_long = rsi_7[i] > 40.0  # Not oversold on fast RSI
        rsi_7_ok_short = rsi_7[i] < 60.0  # Not overbought on fast RSI
        rsi_14_ok_long = rsi_14[i] > 45.0
        rsi_14_ok_short = rsi_14[i] < 55.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TIER 1: ALL HTF ALIGNED + PRIME SESSION (full size)
        # Long: 15m HMA bull + 1h bull + 4h bull + 1d bull + vol ok + RSI ok + prime session
        if hma_bull and htf_1h_bull and htf_4h_bull and htf_1d_bull and vol_ok and rsi_7_ok_long and above_sma200 and prime_session:
            desired_signal = SIZE
        
        # Short: All bear aligned
        elif hma_bear and htf_1h_bear and htf_4h_bear and htf_1d_bear and vol_ok and rsi_7_ok_short and below_sma200 and prime_session:
            desired_signal = -SIZE
        
        # TIER 2: 1h + 4h aligned + PRIME SESSION (80% size)
        # Relax 1d filter for more trades
        elif hma_bull and htf_1h_bull and htf_4h_bull and vol_ok and rsi_7[i] > 45.0 and above_sma200 and prime_session:
            desired_signal = SIZE * 0.8
        
        elif hma_bear and htf_1h_bear and htf_4h_bear and vol_ok and rsi_7[i] < 55.0 and below_sma200 and prime_session:
            desired_signal = -SIZE * 0.8
        
        # TIER 3: 15m + 1h aligned + PRIME SESSION (60% size)
        # Even more relaxed for trade generation
        elif hma_bull and htf_1h_bull and vol_ok and rsi_7[i] > 50.0 and prime_session:
            desired_signal = SIZE * 0.6
        
        elif hma_bear and htf_1h_bear and vol_ok and rsi_7[i] < 50.0 and prime_session:
            desired_signal = -SIZE * 0.6
        
        # TIER 4: Strong 15m momentum + SECONDARY SESSION (40% size)
        # Allow afternoon session for strong signals
        elif hma_bull and rsi_7[i] > 55.0 and vol_ok and rsi_14[i] > 50.0 and secondary_session:
            desired_signal = SIZE * 0.4
        
        elif hma_bear and rsi_7[i] < 45.0 and vol_ok and rsi_14[i] < 50.0 and secondary_session:
            desired_signal = -SIZE * 0.4
        
        # TIER 5: VERY STRONG 15m signal (ignore session, 30% size)
        # Ensures we capture major moves even outside prime hours
        elif hma_bull and rsi_7[i] > 65.0 and vol_ok and above_sma200:
            desired_signal = SIZE * 0.3
        
        elif hma_bear and rsi_7[i] < 35.0 and vol_ok and below_sma200:
            desired_signal = -SIZE * 0.3
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.7:
            final_signal = SIZE * 0.8
        elif desired_signal <= -SIZE * 0.7:
            final_signal = -SIZE * 0.8
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
        elif desired_signal >= SIZE * 0.3:
            final_signal = SIZE * 0.4
        elif desired_signal <= -SIZE * 0.3:
            final_signal = -SIZE * 0.4
        elif desired_signal >= SIZE * 0.1:
            final_signal = SIZE * 0.3
        elif desired_signal <= -SIZE * 0.1:
            final_signal = -SIZE * 0.3
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
                # Flip position
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