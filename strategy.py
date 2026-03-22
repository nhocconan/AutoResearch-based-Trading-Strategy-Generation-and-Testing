#!/usr/bin/env python3
"""
Experiment #075: 1h Primary + 4h/1d HTF — Regime-Adaptive with Connors RSI

Hypothesis: Previous 1h strategies failed due to either (a) 0 trades from over-filtering,
or (b) too many trades (>200/yr) causing fee drag. This strategy uses:

1. CHOPPINESS INDEX (CHOP) for regime detection:
   - CHOP > 55 = ranging market → mean reversion (Connors RSI extremes)
   - CHOP < 45 = trending market → follow HTF HMA direction
   - 45-55 = neutral → reduce position size or skip

2. CONNORS RSI (CRSI) for mean reversion entries:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > SMA(200)
   - Short: CRSI > 90 + price < SMA(200)
   - Proven 75% win rate in range markets

3. HTF HMA(21) on 4h for trend bias:
   - Price > 4h HMA = prefer longs
   - Price < 4h HMA = prefer shorts

4. SESSION FILTER (8-20 UTC):
   - Only trade during high-liquidity hours
   - Reduces false breakouts in Asian session

5. VOLUME FILTER:
   - Volume > 0.8x 20-bar average
   - Confirms genuine moves

6. POSITION SIZING: 0.25 (conservative for 1h)
   - Discrete levels: 0.0, ±0.15, ±0.25
   - Stoploss: 2.0 * ATR(14) trailing

Why this should work:
- Regime detection prevents trend strategies in chop (major failure mode)
- Connors RSI catches reversals at extremes (works in bear/range markets)
- 4h HMA provides trend bias without over-filtering
- Session filter reduces noise (critical for 1h)
- Target: 40-70 trades/year (fee-safe for 1h)

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete
Stoploss: 2.0 * ATR(14) trailing
Target trades: 40-70/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_connors_chop_4h1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = highly choppy/ranging (mean reversion favored)
    - CHOP < 38.2 = trending (trend following favored)
    - 38.2-61.8 = neutral
    """
    atr = calculate_atr(high, low, close, period)
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    price_range = hh - ll
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Components:
    1. RSI(3) on close - short-term momentum
    2. RSI(2) on streak duration - measures consecutive up/down days
    3. PercentRank(100) - where current price ranks vs last 100 closes
    
    Entry signals:
    - CRSI < 10 = oversold (long opportunity)
    - CRSI > 90 = overbought (short opportunity)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak duration
    # Streak = consecutive up or down bars
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to absolute duration for RSI calculation
    # Positive streak = up days, negative = down days
    streak_duration = np.abs(streak)
    streak_direction = np.sign(streak)
    
    # RSI on streak (use streak values directly)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: PercentRank
    # Where does current close rank vs last pr_period closes?
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = close[i-pr_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = rank / pr_period * 100
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    # Convert to hours UTC
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, 200)
    
    # Volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract hour for session filter
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(250, n):  # Start after 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(vol_avg_20[i]):
            continue
        
        # === REGIME DETECTION (CHOP) ===
        chop_value = chop_14[i]
        is_range_regime = chop_value > 55
        is_trend_regime = chop_value < 45
        is_neutral_regime = 45 <= chop_value <= 55
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === SMA(200) FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Slightly relaxed from 10 for more trades
        crsi_overbought = crsi[i] > 85  # Slightly relaxed from 90
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_neutral_regime:
            current_size = REDUCED_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if is_range_regime:
            # Mean reversion in range: CRSI oversold + price above SMA200
            if crsi_oversold and price_above_sma200 and volume_ok and in_session:
                new_signal = current_size
        elif is_trend_regime:
            # Trend following: price above 4h HMA + CRSI not overbought
            if price_above_4h_hma and crsi[i] < 70 and volume_ok and in_session:
                # Enter on pullback (CRSI < 50 in uptrend)
                if crsi[i] < 50:
                    new_signal = current_size
                # Or enter on continuation (CRSI rising from neutral)
                elif 40 < crsi[i] < 60 and crsi[i] > crsi[i-1]:
                    new_signal = REDUCED_SIZE
        
        # SHORT ENTRIES
        if is_range_regime:
            # Mean reversion in range: CRSI overbought + price below SMA200
            if crsi_overbought and price_below_sma200 and volume_ok and in_session:
                new_signal = -current_size
        elif is_trend_regime:
            # Trend following: price below 4h HMA + CRSI not oversold
            if price_below_4h_hma and crsi[i] > 30 and volume_ok and in_session:
                # Enter on pullback (CRSI > 50 in downtrend)
                if crsi[i] > 50:
                    new_signal = -current_size
                # Or enter on continuation (CRSI falling from neutral)
                elif 40 < crsi[i] < 60 and crsi[i] < crsi[i-1]:
                    new_signal = -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~8 days on 1h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if is_range_regime:
                if crsi[i] < 20 and price_above_sma200:
                    new_signal = REDUCED_SIZE
                elif crsi[i] > 80 and price_below_sma200:
                    new_signal = -REDUCED_SIZE
            elif is_trend_regime:
                if price_above_4h_hma and crsi[i] < 45:
                    new_signal = REDUCED_SIZE
                elif price_below_4h_hma and crsi[i] > 55:
                    new_signal = -REDUCED_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME CHANGE EXIT ===
        regime_change_exit = False
        if in_position and position_side != 0:
            # Exit if regime changes against position
            if position_side > 0 and is_trend_regime and price_below_4h_hma:
                regime_change_exit = True
            if position_side < 0 and is_trend_regime and price_above_4h_hma:
                regime_change_exit = True
            # Exit range positions if CRSI returns to neutral
            if is_range_regime:
                if position_side > 0 and crsi[i] > 60:
                    regime_change_exit = True
                if position_side < 0 and crsi[i] < 40:
                    regime_change_exit = True
        
        # Apply stoploss or regime change exit
        if stoploss_triggered or regime_change_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals