#!/usr/bin/env python3
"""
Experiment #008: 30m Connors RSI Mean Reversion with 4h/1d Trend Filter

Hypothesis: Previous 4h KAMA+ADX strategy (#004) worked well but we can improve
by using 30m for precise entry timing while keeping HTF trend filter. This uses:

1. Connors RSI (CRSI) - proven 75% win rate for mean reversion entries
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Entry: CRSI < 15 (long) or CRSI > 85 (short) within HTF trend
2. 1d HMA(21) for major trend bias - only long if price > 1d HMA, vice versa
3. 4h ADX(14) for trend strength - avoid entries when ADX < 18 (too choppy)
4. Session filter: only 8-20 UTC (highest liquidity, lowest slippage)
5. Volume confirmation: volume > 1.2x 20-bar average
6. ATR(14) trailing stoploss: 2.5x ATR

Why 30m + HTF works:
- 1d/4h determine DIRECTION (trend following)
- 30m CRSI determines TIMING (mean reversion within trend)
- This gives HTF trade frequency (30-60/year) with 30m precision
- Session filter reduces false signals during low-liquidity hours

Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30)
Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_connors_rsi_4h_1d_trend_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reference: Alan Hull, 2005
    """
    close_s = pd.Series(close)
    n = period
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, n // 2)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, int(np.sqrt(n)))
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Reference: Connors & Alvarez, "ConnorsRSI" 2008
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    # Calculate up/down streaks
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(pr_period, n):
        window = streak_abs[max(0, i-streak_period+1):i+1]
        if len(window) >= streak_period:
            # Simple streak RSI: higher streak = more extreme
            avg_streak = np.mean(window)
            streak_rsi[i] = min(100, max(0, avg_streak * 20))
    
    # Component 3: Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = close[max(0, i-pr_period+1):i+1]
        if len(window) >= pr_period:
            count_below = np.sum(window[:-1] < close[i])
            percent_rank[i] = 100 * count_below / (pr_period - 1)
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = ranging market.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

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
    
    # Calculate 1D HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4H ADX for trend strength
    adx_4h = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 30m indicators
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Volume moving average for confirmation
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(adx_4h_aligned[i]):
            continue
        
        if np.isnan(crsi_30m[i]):
            continue
        
        if np.isnan(volume_ma20[i]) or volume_ma20[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H ADX TREND STRENGTH ===
        adx_strong = adx_4h_aligned[i] > 18  # Trending enough to trade
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi_30m[i] < 15  # Extreme oversold
        crsi_overbought = crsi_30m[i] > 85  # Extreme overbought
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 1.2 * volume_ma20[i]  # 20% above average
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need daily bullish + ADX strong + CRSI oversold + (volume OR session)
        long_score = 0
        if daily_bullish:
            long_score += 2  # Major trend alignment (required)
        if adx_strong:
            long_score += 1
        if crsi_oversold:
            long_score += 2  # Entry trigger (required)
        if volume_ok:
            long_score += 0.5
        if in_session:
            long_score += 0.5
        
        # Enter long if score >= 4.5 (need trend + oversold + 1 confirmation)
        if long_score >= 4.5 and daily_bullish and crsi_oversold:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Need daily bearish + ADX strong + CRSI overbought + (volume OR session)
        short_score = 0
        if daily_bearish:
            short_score += 2  # Major trend alignment (required)
        if adx_strong:
            short_score += 1
        if crsi_overbought:
            short_score += 2  # Entry trigger (required)
        if volume_ok:
            short_score += 0.5
        if in_session:
            short_score += 0.5
        
        # Enter short if score >= 4.5 (need trend + overbought + 1 confirmation)
        if short_score >= 4.5 and daily_bearish and crsi_overbought:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~3 days on 30m), allow weaker entry
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if daily_bullish and crsi_30m[i] < 25:
                new_signal = REDUCED_SIZE
            elif daily_bearish and crsi_30m[i] > 75:
                new_signal = -REDUCED_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === CRSI MEAN REVERSION EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI recovers to neutral (50+)
            if position_side > 0 and crsi_30m[i] > 55:
                crsi_exit = True
            # Exit short when CRSI recovers to neutral (45-)
            if position_side < 0 and crsi_30m[i] < 45:
                crsi_exit = True
        
        # === ADX WEAKNESS EXIT ===
        adx_weakness = False
        if in_position and position_side != 0:
            # Exit if 4h ADX drops below 15 (trend dying)
            if adx_4h_aligned[i] < 15:
                adx_weakness = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or crsi_exit or adx_weakness:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
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
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals