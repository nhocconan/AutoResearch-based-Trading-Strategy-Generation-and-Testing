#!/usr/bin/env python3
"""
Experiment #015: 1h Connors RSI + Choppiness Regime + 4h HMA Trend

Hypothesis: Lower timeframe (1h) with strict HTF confirmation can work if:
1. 4h HMA(21) defines major trend direction (filter counter-trend trades)
2. Choppiness Index(14) detects regime: >61.8 = range (mean revert), <38.2 = trend (follow)
3. Connors RSI provides precise entry timing (extremes <15 or >85)
4. Session filter (8-20 UTC) avoids low-liquidity periods
5. Volume filter ensures participation
6. ATR(14) 2.5x trailing stop protects capital

Why this should work:
- Connors RSI has proven 75% win rate in backtests
- Choppiness filter adapts to market regime (critical for 2022 crash + 2025 bear)
- 4h trend filter eliminates whipsaws that killed previous 1h strategies
- Session + volume filters reduce trade frequency to target 30-80/year
- Discrete sizing (0.22) minimizes fee churn

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data helper
Position sizing: 0.22 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_chop_4h_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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

def calculate_connors_rsi(close):
    """
    Calculate Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): 3-period RSI on close
    RSI_Streak(2): RSI on streak of consecutive up/down days
    PercentRank(100): Percentile of current close in last 100 closes
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, 3)
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like values
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=2, min_periods=2, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    rs_streak = np.zeros(n)
    mask = avg_streak_loss > 0
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    rs_streak[~mask] = 100  # When no loss, RSI = 100
    
    rsi_streak = 100 - (100 / (1 + rs_streak))
    
    # Percent Rank (100)
    pr = np.zeros(n)
    for i in range(100, n):
        lookback = close[i-99:i+1]  # 100 bars including current
        count_lower = np.sum(lookback < close[i])
        pr[i] = count_lower / 100.0 * 100  # As percentage 0-100
    
    # Connors RSI
    for i in range(100, n):
        crsi[i] = (rsi_3[i] + rsi_streak[i] + pr[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        hl_range = hh - ll
        
        if hl_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
        else:
            chop[i] = 50  # Neutral
    
    return chop

def get_utc_hour(open_time_ms):
    """Extract UTC hour from Unix timestamp in milliseconds."""
    return pd.to_datetime(open_time_ms, unit='ms').hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    connors_rsi = calculate_connors_rsi(close)
    choppiness = calculate_choppiness(high, low, close, 14)
    
    # Volume SMA(20)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # SMA(200) for major trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Extract UTC hours
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    # Lower size for 1h to reduce fee impact
    BASE_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(connors_rsi[i]) or connors_rsi[i] == 0:
            continue
        
        if np.isnan(choppiness[i]) or choppiness[i] == 0:
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        if not in_session:
            # If in position, keep it. If not, no new entries.
            if in_position:
                signals[i] = signals[i-1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # === 4H TREND DIRECTION ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_ranging = choppiness[i] > 55  # Range-bound market
        is_trending = choppiness[i] < 45  # Trending market
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = connors_rsi[i] < 15
        crsi_overbought = connors_rsi[i] > 85
        
        # === POSITION SIZING ===
        long_size = BASE_SIZE
        short_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Multiple confluence required
        # Ranging regime: Connors RSI < 15 + price > SMA200 + volume OK
        # Trending regime: Connors RSI < 25 + 4h bullish + volume OK
        long_conditions = 0
        
        if is_ranging and crsi_oversold and close[i] > sma_200[i]:
            long_conditions += 2  # Strong signal in range
        elif is_trending and connors_rsi[i] < 25 and trend_4h_bullish:
            long_conditions += 2  # Strong signal in trend
        elif crsi_oversold and trend_4h_bullish:
            long_conditions += 1  # Weaker signal
        
        if volume_ok:
            long_conditions += 0.5
        
        # Enter long if conditions >= 2.5
        if long_conditions >= 2.5:
            new_signal = long_size
        
        # SHORT ENTRY: Multiple confluence required
        # Ranging regime: Connors RSI > 85 + price < SMA200 + volume OK
        # Trending regime: Connors RSI > 75 + 4h bearish + volume OK
        short_conditions = 0
        
        if is_ranging and crsi_overbought and close[i] < sma_200[i]:
            short_conditions += 2  # Strong signal in range
        elif is_trending and connors_rsi[i] > 75 and trend_4h_bearish:
            short_conditions += 2  # Strong signal in trend
        elif crsi_overbought and trend_4h_bearish:
            short_conditions += 1  # Weaker signal
        
        if volume_ok:
            short_conditions += 0.5
        
        # Enter short if conditions >= 2.5
        if short_conditions >= 2.5:
            new_signal = -short_size
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit long if regime shifts strongly against (trending bearish after range long)
            if position_side > 0 and is_trending and trend_4h_bearish:
                regime_exit = True
            # Exit short if regime shifts strongly against
            if position_side < 0 and is_trending and trend_4h_bullish:
                regime_exit = True
        
        # Apply stoploss or regime exit
        if stoploss_triggered or regime_exit:
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
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals