#!/usr/bin/env python3
"""
Experiment #125: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: Previous trend-following strategies (#113-#124) failed because BTC/ETH in 2025
is bear/range market, not trending. This strategy adapts by:

1) Choppiness Index (14) regime filter: CHOP>55 = range (mean revert), CHOP<45 = trend
2) Connors RSI for entries: CRSI<10 long, CRSI>90 short (proven 75% win rate)
3) 4h HMA(21) for macro bias: only long if price>4h_HMA, only short if price<4h_HMA
4) Session filter: only trade 8-20 UTC (high liquidity, less whipsaw)
5) Volume confirmation: volume > 0.8x 20-bar avg
6) Tight stoploss: 2.0*ATR trailing

Why this should work:
- CRSI mean reversion excels in range markets (2025 bear/range)
- CHOP filter prevents mean reversion in strong trends
- 4h HTF bias prevents counter-trend trades
- 1h TF with strict filters = 30-60 trades/year (low fee drag)
- Conservative size (0.20-0.25) limits drawdown

Position size: 0.20 base, 0.25 max with volume confluence
Stoploss: 2.0*ATR trailing
Target: 30-60 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, period_rsi)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1] if i > 0 else 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(period_streak, n):
        streak_window = streak[max(0, i-period_streak+1):i+1]
        up_streaks = np.sum(streak_window > 0)
        down_streaks = np.sum(streak_window < 0)
        if up_streaks + down_streaks > 0:
            streak_rsi[i] = 100.0 * up_streaks / (up_streaks + down_streaks)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(period_rank, n):
        returns = np.diff(close[max(0, i-period_rank+1):i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            percent_rank[i] = 100.0 * np.sum(returns <= current_return) / len(returns)
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * (ATR(1) * sqrt(n)) / (Highest High - Lowest Low) over n periods
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high - lowest_low > 0:
            # Calculate ATR(1) over the period (sum of TR / period)
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr1 = high[j] - low[j]
                tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
                tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
                tr = max(tr1, tr2, tr3)
                tr_sum += tr
            atr_sum = tr_sum / period
            
            chop[i] = 100.0 * (atr_sum * np.sqrt(period)) / (highest_high - lowest_low)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hour = pd.to_datetime(open_time, unit='ms').hour
    return hour

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
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for stronger trend filter
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Calculate 1h HMA for additional trend filter
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === HTF TREND BIAS (4h + 1d HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_range = chop_14[i] > 55.0  # Ranging market
        chop_trend = chop_14[i] < 45.0  # Trending market
        
        # === 1h TREND FILTER ===
        hma_1h_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_1h_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Long signal
        crsi_overbought = crsi[i] > 85.0  # Short signal
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 0.8
        volume_strong = volume_ratio > 1.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime: Range (CHOP>55) OR Trend up (CHOP<45 + HTF bullish)
        # CRSI oversold + HTF bias + session + volume
        if crsi_oversold and in_session and volume_confirmed:
            # Range market: mean reversion long
            if chop_range and price_above_hma_4h:
                new_signal = POSITION_SIZE_BASE
                if volume_strong:
                    new_signal = POSITION_SIZE_MAX
            # Trend market: only long if HTF confirms
            elif chop_trend and price_above_hma_4h and price_above_hma_1d:
                new_signal = POSITION_SIZE_BASE
                if volume_strong and hma_1h_bullish:
                    new_signal = POSITION_SIZE_MAX
            # Neutral regime: require strong HTF confirmation
            elif price_above_hma_4h and price_above_hma_1d and hma_1h_bullish:
                new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        # Regime: Range (CHOP>55) OR Trend down (CHOP<45 + HTF bearish)
        # CRSI overbought + HTF bias + session + volume
        if crsi_overbought and in_session and volume_confirmed:
            # Range market: mean reversion short
            if chop_range and price_below_hma_4h:
                new_signal = -POSITION_SIZE_BASE
                if volume_strong:
                    new_signal = -POSITION_SIZE_MAX
            # Trend market: only short if HTF confirms
            elif chop_trend and price_below_hma_4h and price_below_hma_1d:
                new_signal = -POSITION_SIZE_BASE
                if volume_strong and hma_1h_bearish:
                    new_signal = -POSITION_SIZE_MAX
            # Neutral regime: require strong HTF confirmation
            elif price_below_hma_4h and price_below_hma_1d and hma_1h_bearish:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if CRSI not at extreme exit and HTF trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI < 70 and HTF still bullish
                if crsi[i] < 70.0 and price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI > 30 and HTF still bearish
                if crsi[i] > 30.0 and price_below_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON HTF TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals