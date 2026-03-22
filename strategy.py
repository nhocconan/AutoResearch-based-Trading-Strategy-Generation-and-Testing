#!/usr/bin/env python3
"""
Experiment #388: 30m Primary + 4h/1d HTF — Confluence Trend + Session Filter

Hypothesis: Lower TF (30m) strategies fail due to excessive trades → fee drag.
Solution: Use 4h/1d for DIRECTION, 30m only for ENTRY TIMING with strict filters.

Key filters (ALL must align for entry):
1. 1d HMA(21) = major regime (bull/bear bias)
2. 4h HMA(21/50) = intermediate trend confirmation
3. 30m Connors RSI < 15 (long) or > 85 (short) = oversold/overbought entry
4. Session filter: only 8-20 UTC (highest volume, avoids Asia chop)
5. Volume filter: current volume > 0.8x 20-bar average
6. Price vs SMA200: long only if above, short only if below

Why this might work:
- 30m entries within 4h/1d trend = HTF trade frequency with LTF precision
- Session filter removes 2/3 of bars (reduces trades by ~66%)
- Connors RSI extremes = high-probability mean-reversion entries within trend
- 3+ confluence = very selective entries (target 40-70 trades/year)

Position sizing: 0.25 (smaller for lower TF to reduce fee impact)
Stoploss: 2.5 * ATR trailing
Target: 40-70 trades/year, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_confluence_session_crsi_4h1d_v1"
timeframe = "30m"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term momentum
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_s = pd.Series(streak_gain)
    streak_loss_s = pd.Series(streak_loss)
    
    avg_streak_gain = streak_gain_s.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss_s.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100) - where current close ranks vs prior 100 closes
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
    ) * 100
    
    # Combine into CRSI
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_sma(volume, period=20):
    """Calculate volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

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
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, 200)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    # Extract UTC hour for session filter
    utc_hours = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for lower TF)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(sma_200[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 1D MAJOR REGIME (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND (confirmation) ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === PRICE POSITION (SMA200 filter) ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES (entry timing) ===
        # CRSI < 15 = extremely oversold (long entry in uptrend)
        # CRSI > 85 = extremely overbought (short entry in downtrend)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Avoids Asia session chop, trades during high-volume periods
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLUME FILTER ===
        # Current volume must be > 0.8x 20-bar average
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === ENTRY LOGIC - STRICT CONFLUENCE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: All filters must align
        # 1D bull + 4H bull + price>SMA200 + CRSI<15 + session + volume
        if (bull_regime and hma_4h_bullish and price_above_sma200 and 
            crsi_oversold and in_session and volume_ok):
            new_signal = LONG_SIZE
        
        # SHORT ENTRY: All filters must align
        # 1D bear + 4H bear + price<SMA200 + CRSI>85 + session + volume
        if (bear_regime and hma_4h_bearish and price_below_sma200 and 
            crsi_overbought and in_session and volume_ok):
            if new_signal == 0.0:
                new_signal = -SHORT_SIZE
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 40 bars (~20 hours on 30m), relax CRSI threshold slightly
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            crsi_oversold_relaxed = crsi[i] < 25.0
            crsi_overbought_relaxed = crsi[i] > 75.0
            
            if bull_regime and hma_4h_bullish and price_above_sma200 and crsi_oversold_relaxed and in_session:
                new_signal = LONG_SIZE * 0.7
            elif bear_regime and hma_4h_bearish and price_below_sma200 and crsi_overbought_relaxed and in_session:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and crsi[i] > 70:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 30:
            new_signal = 0.0
        
        # Session exit (close position outside trading hours)
        if in_position and not in_session:
            new_signal = 0.0
        
        # Regime flip exit (1D or 4H trend reversal)
        if in_position and position_side > 0 and (bear_regime or hma_4h_bearish):
            new_signal = 0.0
        if in_position and position_side < 0 and (bull_regime or hma_4h_bullish):
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
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
                # Position flip
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