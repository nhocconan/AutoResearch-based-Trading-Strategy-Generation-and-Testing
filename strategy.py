#!/usr/bin/env python3
"""
Experiment #025: 1h Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion

Hypothesis: Lower timeframe strategies fail due to excessive trades and fee drag.
This strategy uses STRICT confluence to limit trades to 30-60/year:

1. 1d HMA(21) for MAJOR trend bias
2. 4h HMA(21) for INTERMEDIATE trend confirmation
3. Connors RSI(3,2,100) for entry timing (extreme <25 or >75)
4. Choppiness Index(14) regime filter (>55 = range, <45 = trend)
5. Volume filter (>0.7x 20-bar avg)
6. Session filter (8-20 UTC only - high liquidity)
7. ATR(14) trailing stoploss at 2.5x

Why this should work:
- 1h entries within 4h/1d trend = HTF frequency with LTF precision
- Connors RSI has 75% win rate for mean reversion
- Choppiness Index prevents trend-following in ranges
- Session filter avoids low-liquidity whipsaws
- Volume confirms genuine moves
- Discrete sizing (0.25) minimizes fee churn

Timeframe: 1h (REQUIRED per experiment #025)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi.fillna(50).values
    
    # Component 3: Percent Rank
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= rank_period else 50
    )
    percent_rank = percent_rank.fillna(50).values
    
    # Connors RSI
    connors_rsi = (rsi_3 + streak_rsi + percent_rank) / 3
    return connors_rsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1) sum / ATR(period)) / (Highest High - Lowest Low) * log10(period)
    
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    
    chop = 100 * (atr1_sum / atr_period) / hh_ll * np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    connors_rsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(connors_rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # If not in session and flat, stay flat (but allow exits for existing positions)
        if not in_session and position_side == 0:
            signals[i] = 0.0
            continue
        
        # === 1D TREND BIAS (MAJOR) ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = connors_rsi[i] < 25
        crsi_overbought = connors_rsi[i] > 75
        
        # Default: hold current position
        new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if position_side != 0:
            if position_side > 0:  # Long
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            else:  # Short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        else:
            # === ENTRY LOGIC (only if flat) ===
            if position_side == 0 and in_session:
                bars_since_last_trade = i - last_trade_bar
                
                # LONG ENTRIES
                if (trend_1d_bullish or is_range) and trend_4h_bullish and volume_ok and crsi_oversold:
                    new_signal = BASE_SIZE
                
                # SHORT ENTRIES
                elif (trend_1d_bearish or is_range) and trend_4h_bearish and volume_ok and crsi_overbought:
                    new_signal = -BASE_SIZE
                
                # === FREQUENCY SAFEGUARD ===
                # If no trades for 200 bars (~8 days on 1h), allow weaker entry
                elif bars_since_last_trade > 200:
                    if trend_1d_bullish and trend_4h_bullish and connors_rsi[i] < 35:
                        new_signal = BASE_SIZE * 0.6
                    elif trend_1d_bearish and trend_4h_bearish and connors_rsi[i] > 65:
                        new_signal = -BASE_SIZE * 0.6
            
            # === TREND REVERSAL EXIT ===
            elif position_side != 0:
                if position_side > 0 and trend_1d_bearish and connors_rsi[i] > 70:
                    new_signal = 0.0
                elif position_side < 0 and trend_1d_bullish and connors_rsi[i] < 30:
                    new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0 and position_side == 0:
            # New entry
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_price = close[i] if position_side > 0 else 0.0
            lowest_price = close[i] if position_side < 0 else 0.0
            last_trade_bar = i
        elif new_signal == 0.0 and position_side != 0:
            # Exit
            position_side = 0
            entry_price = 0.0
            highest_price = 0.0
            lowest_price = 0.0
            last_trade_bar = i
        elif np.sign(new_signal) != position_side and new_signal != 0.0:
            # Reverse
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_price = close[i] if position_side > 0 else 0.0
            lowest_price = close[i] if position_side < 0 else 0.0
            last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals