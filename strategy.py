#!/usr/bin/env python3
"""
Experiment #508: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness + HTF Trend

Hypothesis: After 448 failed strategies, try a DIFFERENT approach for 30m timeframe:

1. CONNORS RSI (CRSI): Proven 75% win rate mean reversion signal
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 15, Short: CRSI > 85
   
2. CHOPPINESS INDEX regime filter: CHOP > 55 = range (mean revert),
   CHOP < 40 = trending (avoid mean reversion). Critical for 30m!
   
3. 4H HMA(21) trend filter: Only trade mean reversion IN trend direction
   Prevents catching falling knives in strong downtrends
   
4. 1D HMA(21) regime filter: Major trend confirmation
   Bull: prefer longs. Bear: prefer shorts + only extreme long entries

5. VOLUME FILTER: volume > 0.6x 20-bar avg (avoid low liquidity traps)

6. SESSION FILTER: 6-22 UTC only (avoid Asian low-liquidity hours)

Why this might beat current best (Sharpe=0.435):
- CRSI is DIFFERENT from simple RSI (448 failed with RSI/Choppiness alone)
- 30m TF with HTF filters = HTF trade frequency, 30m entry precision
- 4-5 confluence filters = target 40-80 trades/year (not 200+)
- Conservative size 0.20 for lower TF (fee drag control)
- 2.5x ATR trailing stop protects in 2022-style crashes

Position sizing: 0.20 (conservative for 30m)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-80 trades/year, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_hma_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    - RSI(3): Short-term momentum
    - RSI_Streak(2): Duration of up/down streak
    - PercentRank(100): Where current price ranks vs last 100 bars
    
    Entry: CRSI < 10-15 (long), CRSI > 85-90 (short)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of streak length (2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1]) * 100.0 if len(x) > 1 else 50.0,
        raw=False
    )
    
    # Combine components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean reversion favorable)
    CHOP < 38.2 = trending market (trend following favorable)
    
    Formula: 100 * LOG10(SUM(High-Low, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Sum of high-low ranges over period
    hl_range = high_s - low_s
    sum_hl = hl_range.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Net movement (absolute difference)
    net_move = np.abs(highest_high - lowest_low)
    net_move = net_move.replace(0, 1e-10)
    
    # Choppiness formula
    chop = 100.0 * np.log10(sum_hl / net_move) / np.log10(period)
    
    return chop.values

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
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_session_filter(open_time):
    """
    Session filter: 6-22 UTC (16 hours, avoids Asian low-liquidity).
    Returns boolean mask.
    """
    # Convert to datetime and extract hour
    hours = pd.to_datetime(open_time, unit='ms').dt.hour.values
    # Active session: 6-22 UTC (16 hours per day)
    session_mask = (hours >= 6) & (hours <= 21)
    return session_mask

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume filter: 20-period average
    vol_s = pd.Series(volume)
    vol_avg_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Session filter (6-22 UTC)
    session_mask = calculate_session_filter(prices['open_time'].values)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, conservative for 30m)
    LONG_SIZE = 0.20
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(500, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === HTF TREND DIRECTION (4h HMA) ===
        bull_trend_4h = close[i] > hma_4h_21_aligned[i]
        bear_trend_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope confirmation
        hma_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 1D REGIME FILTER (major trend) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        range_market = chop[i] > 50.0  # Mean reversion favorable
        strong_range = chop[i] > 61.8  # Very choppy
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0  # Oversold (relaxed for more trades)
        crsi_overbought = crsi[i] > 80.0  # Overbought
        crsi_extreme_low = crsi[i] < 12.0  # Very oversold
        crsi_extreme_high = crsi[i] > 88.0  # Very overbought
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.6 * vol_avg_20[i]
        
        # === SESSION FILTER ===
        session_ok = session_mask[i]
        
        # === ENTRY LOGIC - 4+ CONFLUENCE FILTERS ===
        new_signal = 0.0
        
        # LONG: 4h bull + CRSI oversold + range market + volume + session
        if (bull_trend_4h and crsi_oversold and range_market and volume_ok and session_ok):
            new_signal = LONG_SIZE
        
        # LONG alternative: 1d bull + extreme CRSI + volume (stronger signal)
        elif (bull_regime_1d and crsi_extreme_low and volume_ok and session_ok):
            new_signal = LONG_SIZE
        
        # LONG: Strong range + extreme CRSI + 4h bull (high conviction)
        elif (strong_range and crsi_extreme_low and bull_trend_4h and session_ok):
            new_signal = LONG_SIZE
        
        # SHORT: 4h bear + CRSI overbought + range market + volume + session
        if new_signal == 0.0:
            if (bear_trend_4h and crsi_overbought and range_market and volume_ok and session_ok):
                new_signal = -SHORT_SIZE
            
            # SHORT alternative: 1d bear + extreme CRSI + volume
            elif (bear_regime_1d and crsi_extreme_high and volume_ok and session_ok):
                new_signal = -SHORT_SIZE
            
            # SHORT: Strong range + extreme CRSI + 4h bear
            elif (strong_range and crsi_extreme_high and bear_trend_4h and session_ok):
                new_signal = -SHORT_SIZE
        
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
        
        # === EXIT CONDITIONS ===
        # Exit long on CRSI overbought or trend flip bear
        if in_position and position_side > 0:
            if crsi_overbought or (bear_trend_4h and hma_slope_bear):
                new_signal = 0.0
            # Exit if 1d regime flips strongly bearish
            if bear_regime_1d and hma_1d_21_aligned[i] < hma_4h_21_aligned[i]:
                new_signal = 0.0
        
        # Exit short on CRSI oversold or trend flip bull
        if in_position and position_side < 0:
            if crsi_oversold or (bull_trend_4h and hma_slope_bull):
                new_signal = 0.0
            # Exit if 1d regime flips strongly bullish
            if bull_regime_1d and hma_1d_21_aligned[i] > hma_4h_21_aligned[i]:
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
                # Flip position
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