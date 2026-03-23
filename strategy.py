#!/usr/bin/env python3
"""
Experiment #020: 1h Primary + 4h/12h HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: 1h timeframe with strict HTF filters can achieve 40-80 trades/year while
maintaining edge through Connors RSI (proven 75% win rate in mean reversion).

Key components:
1. 4h HMA(21): Primary trend bias (only trade with HTF trend)
2. 12h ADX(14): Regime detection (ADX>25=trend, ADX<20=range)
3. Connors RSI(3,2,100): Entry timing with extreme thresholds (CRSI<15 long, >85 short)
4. Session filter: Only 8-20 UTC (high volume periods, avoid Asia night noise)
5. Volume confirmation: volume > 0.8x 20-period average
6. ATR(14) stoploss: 2.5*ATR trailing stop

Why this should work:
- 1h primary = enough signals for trade generation (unlike 1d which had 0 trades)
- 4h/12h HTF = strong trend/regime filter reduces whipsaw
- Connors RSI = proven mean reversion edge, works in bear/range markets
- Session + volume filters = only 40-80 trades/year (avoids fee drag)
- Discrete sizing (0.25) = controlled drawdown

Position size: 0.25 (conservative for 1h TF)
Stoploss: 2.5*ATR trailing
Target trades: 40-80/year (strict filters prevent >100)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_adx_session_volume_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of the streak length (consecutive up/down days)
    PercentRank: Percentile rank of 1-period price change over lookback
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of Streak Length
    # Calculate streak length (consecutive up/down)
    returns = close_s.diff()
    streak = np.zeros(n)
    
    for i in range(1, n):
        if returns.iloc[i] > 0:
            if i > 0 and returns.iloc[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif returns.iloc[i] < 0:
            if i > 0 and returns.iloc[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # RSI of streak (use absolute streak for RSI calculation)
    # For up streaks, use positive; for down streaks, we want inverse relationship
    streak_series = pd.Series(streak)
    streak_gain = streak_series.where(streak_series > 0, 0.0)
    streak_loss = -streak_series.where(streak_series < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank of 1-period returns
    pct_change = close_s.pct_change() * 100  # percentage change
    percent_rank = pct_change.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= rank_period else np.nan
    )
    
    # Combine components
    crsi = (rsi_3 + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array / 1000) % 86400) / 3600
    return hours.astype(int)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h ADX for regime detection
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_sma_20 = calculate_sma(volume, period=20)
    
    # Extract UTC hour for session filter
    utc_hour = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25  # Conservative for 1h TF
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Need warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(vol_sma_20[i]):
            continue
        if atr_14[i] == 0 or vol_sma_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === VOLUME FILTER ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_20[i]
        
        # === 4H TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 12H REGIME (ADX) ===
        adx_value = adx_12h_aligned[i]
        is_trending = adx_value > 25.0
        is_ranging = adx_value < 20.0
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        crsi_extreme = crsi_oversold or crsi_overbought
        
        # === ENTRY CONDITIONS (ALL MUST PASS FOR TRADE) ===
        new_signal = 0.0
        
        # Only trade during high-volume session with volume confirmation
        if in_session and volume_confirmed:
            # === LONG ENTRY ===
            # Require: CRSI oversold + 4h trend bullish OR ranging regime
            if crsi_oversold:
                if price_above_hma_4h:  # Trend-following long
                    new_signal = POSITION_SIZE
                elif is_ranging:  # Mean reversion in range
                    new_signal = POSITION_SIZE
            
            # === SHORT ENTRY ===
            # Require: CRSI overbought + 4h trend bearish OR ranging regime
            elif crsi_overbought:
                if price_below_hma_4h:  # Trend-following short
                    new_signal = -POSITION_SIZE
                elif is_ranging:  # Mean reversion in range
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, hold unless stoploss or exit signal
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON HTF TREND REVERSAL ===
        # Exit long if 4h trend turns bearish strongly
        if in_position and position_side > 0:
            if price_below_hma_4h and is_trending:
                new_signal = 0.0
        
        # Exit short if 4h trend turns bullish strongly
        if in_position and position_side < 0:
            if price_above_hma_4h and is_trending:
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