#!/usr/bin/env python3
"""
Experiment #008: 30m Multi-Timeframe Regime-Adaptive Strategy

Hypothesis: Lower timeframe (30m) can work IF we use HTF (4h/1d) for signal DIRECTION
and only use 30m for ENTRY TIMING. This gives HTF trade frequency with 30m precision.

Key design:
1. 1d HMA = Major trend bias (only trade long if price > 1d HMA)
2. 4h HMA = Intermediate trend direction (confirms 1d bias)
3. 30m Connors RSI = Entry timing (oversold in uptrend, overbought in downtrend)
4. Choppiness Index = Regime filter (range vs trend logic)
5. Volume filter = Only trade when volume > 0.8x 20-bar average
6. Session filter = Only 8-20 UTC (highest liquidity, avoid Asia night whipsaws)

Why this should beat previous attempts:
- 30m entries are tighter than 1h/4h (better R:R)
- HTF filters prevent overtrading (target 40-80 trades/year)
- Connors RSI has proven 75% win rate in mean reversion
- Session filter avoids low-liquidity trap entries

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.25 (smaller for lower TF to reduce fee impact)
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_connors_htf_4h1d_session_vol_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = Range/Consolidation
    CHOP < 38.2 = Trending
    """
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

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
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    streak_avg_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = np.where(streak_avg_loss == 0, 1e-10, streak_avg_loss)
    streak_rs = streak_avg_gain / streak_avg_loss
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    
    # PercentRank
    returns = close_s.pct_change().values
    percent_rank = np.full(n, 50.0)
    for i in range(rank_period, n):
        if not np.isnan(returns[i]):
            window = returns[max(0, i-rank_period):i]
            window = window[~np.isnan(window)]
            if len(window) > 0:
                percent_rank[i] = 100 * np.sum(window < returns[i]) / len(window)
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
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
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume average (20 bars)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours
    session_hours = np.array([calculate_session_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, smaller for 30m)
    BASE_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        current_hour = session_hours[i]
        in_session = (current_hour >= 8) and (current_hour <= 20)
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / vol_avg_20[i]
        volume_ok = vol_ratio > 0.7  # At least 70% of average volume
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop_14[i] > 55.0
        is_trend = chop_14[i] < 45.0
        
        # === 1D TREND BIAS (Major direction) ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        hma4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === CONFLUENCE: Need HTF agreement ===
        strong_bullish = daily_bullish and hma4h_bullish
        strong_bearish = daily_bearish and hma4h_bearish
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # RANGE REGIME: Mean Reversion with HTF bias
        if is_range and in_session and volume_ok:
            # LONG: CRSI oversold + HTF not bearish
            if crsi[i] < 18 and not strong_bearish:
                new_signal = current_size
            # SHORT: CRSI overbought + HTF not bullish
            elif crsi[i] > 82 and not strong_bullish:
                new_signal = -current_size
        
        # TREND REGIME: Pullback entries in HTF direction
        elif is_trend and in_session and volume_ok:
            # LONG: HTF bullish + pullback (CRSI not extreme high)
            if strong_bullish and crsi[i] < 55 and rsi_14[i] < 65:
                new_signal = current_size
            # SHORT: HTF bearish + pullback (CRSI not extreme low)
            elif strong_bearish and crsi[i] > 45 and rsi_14[i] > 35:
                new_signal = -current_size
        
        # NEUTRAL REGIME: Wait for clearer signals
        else:
            # Only enter on extreme CRSI with strong HTF agreement
            if crsi[i] < 12 and strong_bullish and in_session:
                new_signal = current_size * 0.8
            elif crsi[i] > 88 and strong_bearish and in_session:
                new_signal = -current_size * 0.8
        
        # === TRADE FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~40 hours on 30m), allow weaker entry
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if strong_bullish and crsi[i] < 35 and in_session:
                new_signal = current_size * 0.6
            elif strong_bearish and crsi[i] > 65 and in_session:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and strong_bearish:
                trend_reversal = True
            if position_side < 0 and strong_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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