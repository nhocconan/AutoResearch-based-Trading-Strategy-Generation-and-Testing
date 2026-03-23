#!/usr/bin/env python3
"""
Experiment #405: 1h Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Combining Choppiness Index regime detection with Connors RSI (proven 75% win rate)
and HTF HMA trend bias will beat Sharpe=0.612 on 1h timeframe with controlled trade frequency.

Key innovations vs failed 1h strategies (#395, #398, #400 with Sharpe=0.000):
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven mean reversion
2. Choppiness Index regime: CHOP > 55 = range (use CRSI), CHOP < 45 = trend (use HMA)
3. 4h HMA(21) for HTF bias — only trade long if 4h HMA bullish, short if bearish
4. 1d HMA(21) for overall market bias filter
5. Session filter: only 8-20 UTC (high liquidity hours)
6. Volume filter: volume > 0.7x 20-bar average
7. Discrete position sizing: 0.0, ±0.22 (conservative for 1h TF)
8. ATR(14) trailing stoploss: 2.5x for longs, 2.0x for shorts

Why this should beat Sharpe=0.612:
- CRSI proven in research notes (75% win rate, works in bear/range markets)
- CHOP regime filter proven in #394 (ETH Sharpe +0.923 with CHOP+CRSI)
- 1h TF with strict filters = target 40-70 trades/year = ~2-3.5% fee drag
- Different signal combination than all failed 1h attempts
- Less strict than #395/#398 (which got 0 trades) — multiple entry paths

Target: Sharpe > 0.612, 40-70 trades/year, DD < -40%, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_crsi_hma_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Very short-term momentum
    RSI_Streak(2): Measures consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 bars
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measure consecutive up/down bars
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    # RSI of streak (period 2)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    
    # Percent Rank - where current close ranks vs last rank_period bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # CRSI = average of three components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        # Calculate ATR sum over period
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

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
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21_1h = calculate_hma(close, 21)
    hma_50_1h = calculate_hma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Volume SMA for filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for bias (4h and 1d)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22  # 22% position size for 1h (conservative, target 40-70 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(hma_21_1h[i]) or np.isnan(hma_50_1h[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_sma_20[i] + 1e-10)
        vol_ok = vol_ratio > 0.7  # At least 70% of average volume
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market (use mean reversion)
        is_trending = chop[i] < 45.0  # Trend market (use trend following)
        # Neutral zone: 45 <= CHOP <= 55
        
        # === HTF BIAS (4h and 1d HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (1h HMA crossover) ===
        hma_bullish = hma_21_1h[i] > hma_50_1h[i]
        hma_bearish = hma_21_1h[i] < hma_50_1h[i]
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Extremely oversold
        crsi_overbought = crsi[i] > 85.0  # Extremely overbought
        crsi_neutral = 30.0 < crsi[i] < 70.0  # Neutral zone
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Multiple confluence paths (need 3+ filters agreeing)
        long_bias = price_above_hma_4h and price_above_hma_1d  # HTF bullish
        
        if long_bias and in_session and vol_ok:
            confluence_count = 0
            
            # Path 1: Choppy regime + CRSI oversold (mean reversion)
            if is_choppy and crsi_oversold:
                confluence_count += 2  # Strong signal in range market
            
            # Path 2: Trending regime + HMA bullish + CRSI not overbought
            if is_trending and hma_bullish and crsi[i] < 75.0:
                confluence_count += 1
            
            # Path 3: Price above SMA200 + CRSI pullback
            if price_above_sma200 and 20.0 < crsi[i] < 50.0:
                confluence_count += 1
            
            # Path 4: HMA crossover bullish + CRSI rising from oversold
            if hma_bullish and crsi[i] > 25.0 and crsi_oversold:
                confluence_count += 1
            
            # Need at least 2 confluence factors for entry
            if confluence_count >= 2:
                desired_signal = BASE_SIZE
        
        # SHORT SETUP - Multiple confluence paths
        short_bias = price_below_hma_4h and price_below_hma_1d  # HTF bearish
        
        if short_bias and in_session and vol_ok:
            confluence_count = 0
            
            # Path 1: Choppy regime + CRSI overbought (mean reversion)
            if is_choppy and crsi_overbought:
                confluence_count += 2  # Strong signal in range market
            
            # Path 2: Trending regime + HMA bearish + CRSI not oversold
            if is_trending and hma_bearish and crsi[i] > 25.0:
                confluence_count += 1
            
            # Path 3: Price below SMA200 + CRSI rally
            if price_below_sma200 and 50.0 < crsi[i] < 80.0:
                confluence_count += 1
            
            # Path 4: HMA crossover bearish + CRSI falling from overbought
            if hma_bearish and crsi[i] < 75.0 and crsi_overbought:
                confluence_count += 1
            
            # Need at least 2 confluence factors for entry
            if confluence_count >= 2:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Asymmetric: tighter on shorts) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr  # 2.5x for longs
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr  # 2.0x for shorts
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXIT (extreme reached - take profit) ===
        if in_position and position_side > 0 and crsi_overbought:
            # Long exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_oversold:
            # Short exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_hma_4h:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_4h:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and long_bias:
                desired_signal = BASE_SIZE
            elif position_side < 0 and short_bias:
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