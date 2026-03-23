#!/usr/bin/env python3
"""
Experiment #448: 30m Primary + 4h/1d HTF — Strict Confluence Mean Reversion

Hypothesis: Lower TF (30m) can work IF we use VERY strict entry filters to limit
trades to 30-80/year. Key insight from failures: 30m strategies fail due to
fee drag from too many trades. Solution: 4h/1d for DIRECTION, 30m only for
ENTRY TIMING within HTF trend.

New approach combining proven elements:
1. 4h HMA(21) for primary trend bias (long only when price > 4h HMA)
2. 1d Choppiness Index for regime (CHOP>55 = range/mean-revert, CHOP<45 = trend)
3. Connors RSI(3,2,100) for entry timing (extreme <15 or >85 only)
4. Session filter: only enter 8-20 UTC (high liquidity, less whipsaw)
5. Volume confirmation: taker_buy_ratio >0.55 for longs, <0.45 for shorts
6. ATR(14) volatility filter: skip entries when ATR ratio >2.0 (too volatile)
7. Position size: 0.20 base (smaller for lower TF), discrete levels only
8. Stoploss: 2.5x ATR trailing stop via signal→0

Target: Sharpe >0.612, 40-80 trades/year, DD < -30%
Timeframe: 30m (uses 4h/1d HTF for bias)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_strict_confluence_crsi_chop_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI) = (RSI3 + RSI_Streak2 + PercentRank100) / 3."""
    n = len(close)
    
    # RSI(3) component
    rsi = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank component
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate taker buy ratio for volume confirmation
    taker_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    # Calculate session hours
    session_hours = calculate_session_hour(open_time)
    
    # Calculate and align HTF HMA for bias (4h and 1d)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # 20% base position size for 30m (smaller due to more trades)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= session_hours[i] <= 20
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop[i] > 55.0  # Range market → mean revert
        regime_trend = chop[i] < 45.0  # Trending market → trend follow
        
        # === HTF TREND BIAS (4h + 1d HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias when 4h and 1d agree
        htf_bullish = price_above_hma_4h and price_above_hma_1d
        htf_bearish = price_below_hma_4h and price_below_hma_1d
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === CRSI EXTREME SIGNALS (Very strict thresholds) ===
        crsi_extreme_oversold = crsi[i] < 15.0  # Only extreme oversold
        crsi_extreme_overbought = crsi[i] > 85.0  # Only extreme overbought
        
        # === VOL FILTER (Skip when too volatile) ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        vol_normal = vol_ratio < 2.0  # Skip if ATR > 2x median
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = taker_ratio[i] > 0.55
        volume_bearish = taker_ratio[i] < 0.45
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        confluence_count = 0
        
        # === REGIME 1: RANGE (CHOP > 55) — MEAN REVERSION ===
        if regime_range:
            # Long: CRSI extreme oversold + HTF not bearish + session + volume + vol normal
            if crsi_extreme_oversold and not htf_bearish and in_session and volume_bullish and vol_normal:
                confluence_count = 1
                if price_above_hma_4h:  # 4h bullish adds confidence
                    confluence_count += 1
                if crsi[i] < 10.0:  # Even more extreme
                    confluence_count += 1
                
                if confluence_count >= 2:  # Need at least 2 confluence factors
                    desired_signal = BASE_SIZE
            
            # Short: CRSI extreme overbought + HTF not bullish + session + volume + vol normal
            if crsi_extreme_overbought and not htf_bullish and in_session and volume_bearish and vol_normal:
                if desired_signal == 0:
                    confluence_count = 1
                    if price_below_hma_4h:  # 4h bearish adds confidence
                        confluence_count += 1
                    if crsi[i] > 90.0:  # Even more extreme
                        confluence_count += 1
                    
                    if confluence_count >= 2:
                        desired_signal = -BASE_SIZE
        
        # === REGIME 2: TREND (CHOP < 45) — TREND FOLLOW ===
        elif regime_trend:
            # Long: HTF bullish + pullback (CRSI < 40) + session + volume
            if htf_bullish and crsi[i] < 40.0 and in_session and volume_bullish and vol_normal:
                confluence_count = 1
                if price_above_hma_4h and price_above_hma_1d:
                    confluence_count += 1
                if crsi[i] < 30.0:
                    confluence_count += 1
                
                if confluence_count >= 2:
                    desired_signal = BASE_SIZE
            
            # Short: HTF bearish + rally (CRSI > 60) + session + volume
            if htf_bearish and crsi[i] > 60.0 and in_session and volume_bearish and vol_normal:
                if desired_signal == 0:
                    confluence_count = 1
                    if price_below_hma_4h and price_below_hma_1d:
                        confluence_count += 1
                    if crsi[i] > 70.0:
                        confluence_count += 1
                    
                    if confluence_count >= 2:
                        desired_signal = -BASE_SIZE
        
        # === REGIME 3: TRANSITION (45-55) — NO TRADES ===
        # Stay flat during unclear regimes
        
        # === CAP SIGNAL TO MAX 0.30 ===
        if desired_signal > 0.30:
            desired_signal = 0.30
        elif desired_signal < -0.30:
            desired_signal = -0.30
        
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
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_4h:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_hma_4h:
                desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.25:
                    desired_signal = 0.25
                elif desired_signal >= 0.15:
                    desired_signal = 0.20
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.25:
                    desired_signal = -0.25
                elif desired_signal <= -0.15:
                    desired_signal = -0.20
                else:
                    desired_signal = -0.15
        
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